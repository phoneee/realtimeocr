import CoreGraphics
import Foundation

/// 8-bit RGBA pixels, row 0 = top. The one image type the OCR steps work on.
public struct RGBAImage {
    public let width: Int
    public let height: Int
    public let pixels: [UInt8]

    public enum Failure: Error {
        case emptyImage
        case sizeMismatch
        case bitmapContext
    }

    public init(width: Int, height: Int, pixels: [UInt8]) throws {
        guard width > 0, height > 0 else { throw Failure.emptyImage }
        guard pixels.count == width * height * 4 else { throw Failure.sizeMismatch }
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Draws the image into sRGB RGBA8.
    public init(_ image: CGImage) throws {
        let w = image.width
        let h = image.height
        guard w > 0, h > 0 else { throw Failure.emptyImage }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard let context = CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            ) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw Failure.bitmapContext }
        try self.init(width: w, height: h, pixels: rgba)
    }
}

enum ModelFile {
    static func url(_ name: String, _ ext: String) throws -> URL {
        guard let url = Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Model") else {
            throw LineRecognizer.Failure.missingResource("\(name).\(ext)")
        }
        return url
    }
}
