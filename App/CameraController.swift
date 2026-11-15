import AVFoundation
import CoreImage
import OCRCore
import UIKit

/// Camera frames -> guide-box crop -> LineRecognizer. One job at a time, latest frame only:
/// work runs on the frame queue, and AVFoundation drops frames that arrive while it is busy.
final class CameraController: NSObject, ObservableObject {
    enum Rate: String, CaseIterable, Identifiable {
        case two = "2/s", five = "5/s", max = "Max"

        var id: Self { self }

        var minInterval: CFTimeInterval {
            switch self {
            case .two: 0.5
            case .five: 0.2
            case .max: 0
            }
        }
    }

    struct Status {
        var reading: LineReading?
        var cropMs = 0.0
        var totalMs = 0.0
        var cropSize = CGSize.zero
        var crop: UIImage?
    }

    enum SetupError: LocalizedError {
        case cameraDenied, noCamera, cannotAttach

        var errorDescription: String? {
            switch self {
            case .cameraDenied: "Camera access is off. Allow it in Settings → Thai OCR."
            case .noCamera: "No back camera found."
            case .cannotAttach: "Could not attach the camera to the capture session."
            }
        }
    }

    @Published private(set) var stableText = ""
    @Published private(set) var status = Status()
    @Published private(set) var errorMessage: String?
    @Published var isFrozen = false {
        didSet { let frozen = isFrozen; queue.async { self.frozen = frozen } }
    }
    @Published var rate = Rate.two {
        didSet { let interval = rate.minInterval; queue.async { self.minInterval = interval } }
    }

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "ocr.frames")
    private let ciContext = CIContext()

    // Touched only on `queue`.
    private var recognizer: LineRecognizer?
    private var guide = CGRect.zero // normalized, unrotated sensor picture, top-left origin
    private var frozen = false
    private var minInterval = Rate.two.minInterval
    private var lastRun: CFTimeInterval = 0
    private var candidate: String?

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return self.publish(SetupError.cameraDenied) }
            self.queue.async { self.configureAndRun() }
        }
    }

    /// `rect` from AVCaptureVideoPreviewLayer.metadataOutputRectConverted(fromLayerRect:).
    func setGuide(_ rect: CGRect) {
        queue.async { self.guide = rect }
    }

    private func configureAndRun() {
        do {
            recognizer = try LineRecognizer()
            try configureSession()
            session.startRunning()
        } catch {
            publish(error)
        }
    }

    private func configureSession() throws {
        guard let device = Self.backCamera() else { throw SetupError.noCamera }
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        let input = try AVCaptureDeviceInput(device: device)
        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddInput(input), session.canAddOutput(output) else { throw SetupError.cannotAttach }
        session.addInput(input)
        session.addOutput(output)
        // Small print needs pixels: the crop is a thin strip of the frame.
        // Checked after the input is attached, since what is supported depends on the camera.
        session.sessionPreset = session.canSetSessionPreset(.hd4K3840x2160) ? .hd4K3840x2160 : .hd1920x1080

        try device.lockForConfiguration()
        defer { device.unlockForConfiguration() }
        // Multi-camera devices start on the ultra-wide. Zoom to the wide lens; the system still
        // switches to the ultra-wide macro when the phone gets closer than the wide lens can focus.
        if let wide = device.virtualDeviceSwitchOverVideoZoomFactors.first {
            device.videoZoomFactor = CGFloat(truncating: wide)
        }
        if device.isFocusModeSupported(.continuousAutoFocus) {
            device.focusMode = .continuousAutoFocus
        }
    }

    private static func backCamera() -> AVCaptureDevice? {
        let types: [AVCaptureDevice.DeviceType] = [.builtInTripleCamera, .builtInDualWideCamera, .builtInWideAngleCamera]
        return types.lazy.compactMap { AVCaptureDevice.default($0, for: .video, position: .back) }.first
    }

    /// The guide box cut from the frame and turned upright (back camera, portrait UI).
    private func cropGuide(from pixels: CVPixelBuffer) -> CGImage? {
        let frame = CIImage(cvPixelBuffer: pixels)
        let w = frame.extent.width
        let h = frame.extent.height
        let rect = CGRect(x: guide.minX * w, y: (1 - guide.maxY) * h, width: guide.width * w, height: guide.height * h)
            .integral // Core Image's origin is bottom-left
        guard rect.width >= 1, rect.height >= 1 else { return nil }
        let upright = frame.cropped(to: rect).oriented(.right)
        return ciContext.createCGImage(upright, from: upright.extent)
    }

    /// Two acceptable readings in a row that agree; anything else resets the run.
    private func confirm(_ reading: LineReading) -> String? {
        guard reading.isAcceptable else {
            candidate = nil
            return nil
        }
        defer { candidate = reading.text }
        return reading.text == candidate ? reading.text : nil
    }

    private func publish(_ error: Error) {
        let message = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
        DispatchQueue.main.async { self.errorMessage = message }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let start = CACurrentMediaTime()
        guard !frozen, let recognizer, !guide.isEmpty, start - lastRun >= minInterval,
              let pixels = sampleBuffer.imageBuffer, let crop = cropGuide(from: pixels)
        else { return }
        lastRun = start
        let cropMs = (CACurrentMediaTime() - start) * 1000
        do {
            let reading = try recognizer.read(crop)
            let confirmed = confirm(reading)
            let status = Status(
                reading: reading, cropMs: cropMs, totalMs: (CACurrentMediaTime() - start) * 1000,
                cropSize: CGSize(width: crop.width, height: crop.height), crop: UIImage(cgImage: crop)
            )
            DispatchQueue.main.async {
                guard !self.isFrozen else { return }
                self.errorMessage = nil // a read worked again; setup errors never get here (no frames)
                self.status = status
                if let confirmed, confirmed != self.stableText {
                    self.stableText = confirmed
                }
            }
        } catch {
            publish(error)
        }
    }
}
