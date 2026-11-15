import AVFoundation
import SwiftUI
import UIKit

/// Live preview with the line guide drawn on it. The guide's frame is converted to the
/// camera's own coordinates here, where the preview's crop and scaling are known.
struct CameraPreview: UIViewRepresentable {
    let camera: CameraController
    /// Passed by value so SwiftUI sees the change and calls updateUIView.
    let isFrozen: Bool

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = camera.session
        view.previewLayer.videoGravity = .resizeAspectFill
        view.onGuideChange = camera.setGuide
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        view.previewLayer.connection?.isEnabled = !isFrozen // a disabled connection holds the last frame
    }
}

final class PreviewView: UIView {
    static let guideHeight: CGFloat = 72
    static let guideInset: CGFloat = 20
    /// Guide centre as a share of the view height, clear of the bottom panel.
    static let guideCentre: CGFloat = 0.32

    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    var onGuideChange: ((CGRect) -> Void)?

    private let guideLayer = CAShapeLayer()

    override init(frame: CGRect) {
        super.init(frame: frame)
        guideLayer.strokeColor = UIColor.systemYellow.cgColor
        guideLayer.fillColor = UIColor.clear.cgColor
        guideLayer.lineWidth = 3
        layer.addSublayer(guideLayer)
        // The conversion below needs the session's input; redo it once the camera is running.
        NotificationCenter.default.addObserver(
            self, selector: #selector(sessionStarted), name: AVCaptureSession.didStartRunningNotification, object: nil
        )
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    @objc private func sessionStarted() {
        DispatchQueue.main.async { self.setNeedsLayout() }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        let guide = CGRect(
            x: Self.guideInset, y: bounds.height * Self.guideCentre - Self.guideHeight / 2,
            width: bounds.width - 2 * Self.guideInset, height: Self.guideHeight
        )
        guideLayer.frame = bounds
        guideLayer.path = UIBezierPath(roundedRect: guide, cornerRadius: 8).cgPath
        onGuideChange?(previewLayer.metadataOutputRectConverted(fromLayerRect: guide))
    }
}
