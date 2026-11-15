import AVFoundation
import CoreImage
import OCRCore
import UIKit

/// Camera frames -> whole-frame OCR (PageReader). One job at a time, latest frame only:
/// work runs on the frame queue, and AVFoundation drops frames that arrive while it is busy.
final class CameraController: NSObject, ObservableObject {
    /// The detector's long side: speed against small text.
    enum Detail: Int, CaseIterable, Identifiable {
        case fast = 736, medium = 960, fine = 1280

        var id: Self { self }

        var label: String {
            switch self {
            case .fast: "Fast"
            case .medium: "Medium"
            case .fine: "Fine"
            }
        }
    }

    struct Status {
        var reading: PageReading?
        var frameSize = CGSize.zero
        var renderMs = 0.0
        var totalMs = 0.0
    }

    enum SetupError: LocalizedError {
        case cameraDenied, noCamera, cannotAttach, cannotRender

        var errorDescription: String? {
            switch self {
            case .cameraDenied: "Camera access is off. Allow it in Settings → Thai OCR."
            case .noCamera: "No back camera found."
            case .cannotAttach: "Could not attach the camera to the capture session."
            case .cannotRender: "Could not convert a camera frame."
            }
        }
    }

    @Published private(set) var status = Status()
    @Published private(set) var errorMessage: String?
    @Published var isFrozen = false {
        didSet { let frozen = isFrozen; queue.async { self.frozen = frozen } }
    }
    @Published var detail = Detail.medium {
        didSet { let limit = detail.rawValue; queue.async { self.limit = limit } }
    }

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "ocr.frames")
    private let ciContext = CIContext()

    // Touched only on `queue`.
    private var reader: PageReader?
    private var frozen = false
    private var limit = Detail.medium.rawValue

    private var started = false // main thread

    func start() {
        guard !started else { return } // onAppear can fire again; the session takes one input
        started = true
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else { return self.publish(SetupError.cameraDenied) }
            self.queue.async { self.configureAndRun() }
        }
    }

    private func configureAndRun() {
        do {
            reader = try PageReader()
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
        // Small print needs pixels: lines are cut from the full-resolution frame.
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

    /// The frame turned upright (back camera, portrait UI), as RGBA.
    // ponytail: via CGImage, the path the tests cover. Render straight to a bitmap if this shows up in timings.
    private func uprightFrame(_ pixels: CVPixelBuffer) throws -> RGBAImage {
        let upright = CIImage(cvPixelBuffer: pixels).oriented(.right)
        guard let image = ciContext.createCGImage(upright, from: upright.extent) else { throw SetupError.cannotRender }
        return try RGBAImage(image)
    }

    private func publish(_ error: Error) {
        let message = (error as? LocalizedError)?.errorDescription ?? String(describing: error)
        DispatchQueue.main.async { self.errorMessage = message }
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !frozen, let reader, let pixels = sampleBuffer.imageBuffer else { return }
        let start = CACurrentMediaTime()
        do {
            let frame = try uprightFrame(pixels)
            let renderMs = (CACurrentMediaTime() - start) * 1000
            let reading = try reader.read(frame, limit: limit)
            let status = Status(
                reading: reading, frameSize: CGSize(width: frame.width, height: frame.height),
                renderMs: renderMs, totalMs: (CACurrentMediaTime() - start) * 1000
            )
            DispatchQueue.main.async {
                guard !self.isFrozen else { return }
                self.errorMessage = nil // a read worked again; setup errors never get here (no frames)
                self.status = status
            }
        } catch {
            publish(error)
        }
    }
}
