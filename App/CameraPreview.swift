import AVFoundation
import OCRCore
import SwiftUI
import UIKit

/// Live preview with the detected line boxes drawn over it.
struct CameraPreview: UIViewRepresentable {
    let camera: CameraController
    /// Passed by value so SwiftUI sees changes and calls updateUIView.
    let isFrozen: Bool
    let boxes: [Quad]
    let frameSize: CGSize

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = camera.session
        view.previewLayer.videoGravity = .resizeAspectFill
        return view
    }

    func updateUIView(_ view: PreviewView, context: Context) {
        view.previewLayer.connection?.isEnabled = !isFrozen // a disabled connection holds the last frame
        view.show(boxes, frameSize: frameSize)
    }
}

final class PreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

    private let boxLayer = CAShapeLayer()
    private var boxes: [Quad] = []
    private var frameSize = CGSize.zero

    override init(frame: CGRect) {
        super.init(frame: frame)
        boxLayer.strokeColor = UIColor.systemGreen.cgColor
        boxLayer.fillColor = UIColor.clear.cgColor
        boxLayer.lineWidth = 2
        layer.addSublayer(boxLayer)
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    func show(_ boxes: [Quad], frameSize: CGSize) {
        self.boxes = boxes
        self.frameSize = frameSize
        setNeedsLayout()
    }

    /// Frame pixels -> view points, matching the preview's aspect fill.
    override func layoutSubviews() {
        super.layoutSubviews()
        boxLayer.frame = bounds
        guard frameSize.width > 0, frameSize.height > 0 else { return }
        let scale = max(bounds.width / frameSize.width, bounds.height / frameSize.height)
        let offset = CGPoint(x: (bounds.width - frameSize.width * scale) / 2, y: (bounds.height - frameSize.height * scale) / 2)
        let path = UIBezierPath()
        for quad in boxes {
            let points = quad.map { CGPoint(x: offset.x + $0.x * scale, y: offset.y + $0.y * scale) }
            path.move(to: points[0])
            points.dropFirst().forEach(path.addLine(to:))
            path.close()
        }
        boxLayer.path = path.cgPath
    }
}
