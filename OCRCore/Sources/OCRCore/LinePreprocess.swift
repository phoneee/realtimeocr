import CoreGraphics
import Foundation

/// Port of the recognizer input path in runtime/line_engine.py with `"color": "flatten"`:
/// resize to 48 px high (bilinear, as cv2.INTER_LINEAR) -> BT.601 luma -> `_flatten`
/// -> (x / 255 - 0.5) / 0.5, grey replicated over 3 channels.
enum LinePreprocess {
    static let height = 48
    static let minWidth = 16
    static let maxWidth = 3200

    /// `[1, 3, height, width]`, channel-major.
    struct Tensor {
        let values: [Float]
        let width: Int
    }

    enum Failure: Error {
        case emptyImage
        case bitmapContext
    }

    static func tensor(from image: CGImage) throws -> Tensor {
        let (gray, w, h) = try luma(image)
        let lineWidth = min(maxWidth, max(1, Int((Double(height) * Double(w) / Double(h)).rounded(.up))))
        let small = resizeBilinear(gray, width: w, height: h, toWidth: lineWidth, toHeight: height)
        let flat = flatten(small, width: lineWidth, height: height)

        let width = max(minWidth, lineWidth)
        let plane = height * width
        var values = [Float](repeating: 0, count: 3 * plane) // 0 = mid-grey after normalising (padding)
        for y in 0..<height {
            for x in 0..<lineWidth {
                values[y * width + x] = Float(flat[y * lineWidth + x]) / 127.5 - 1
            }
        }
        for channel in 1..<3 {
            values.replaceSubrange(channel * plane..<(channel + 1) * plane, with: values[0..<plane])
        }
        return Tensor(values: values, width: width)
    }

    /// Luma per pixel, row 0 = top, as cv2.COLOR_BGR2GRAY weights.
    static func luma(_ image: CGImage) throws -> (pixels: [Float], width: Int, height: Int) {
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
        let pixels = (0..<(w * h)).map { i in
            0.299 * Float(rgba[4 * i]) + 0.587 * Float(rgba[4 * i + 1]) + 0.114 * Float(rgba[4 * i + 2])
        }
        return (pixels, w, h)
    }

    /// cv2.resize INTER_LINEAR: half-pixel centres, two taps per axis, no antialiasing.
    static func resizeBilinear(_ src: [Float], width w: Int, height h: Int, toWidth dw: Int, toHeight dh: Int) -> [UInt8] {
        func taps(_ d: Int, scale: Double, size: Int) -> (Int, Int, Float) {
            let f = (Double(d) + 0.5) * scale - 0.5
            var i = Int(f.rounded(.down))
            var t = Float(f - Double(i))
            if i < 0 { i = 0; t = 0 }
            if i >= size - 1 { i = size - 1; t = 0 }
            return (i, min(i + 1, size - 1), t)
        }
        let xTaps = (0..<dw).map { taps($0, scale: Double(w) / Double(dw), size: w) }
        var out = [UInt8](repeating: 0, count: dw * dh)
        for y in 0..<dh {
            let (y0, y1, ty) = taps(y, scale: Double(h) / Double(dh), size: h)
            for x in 0..<dw {
                let (x0, x1, tx) = xTaps[x]
                let top = src[y0 * w + x0] * (1 - tx) + src[y0 * w + x1] * tx
                let bottom = src[y1 * w + x0] * (1 - tx) + src[y1 * w + x1] * tx
                out[y * dw + x] = UInt8(clamping: Int((top * (1 - ty) + bottom * ty).rounded()))
            }
        }
        return out
    }

    /// `_flatten`: dark text on white from any line crop. Each column's background is the
    /// median of the crop's top and bottom margin rows, smoothed along the line; text is
    /// whatever differs from it, light or dark, stretched so the 98th percentile is black.
    static func flatten(_ g: [UInt8], width w: Int, height h: Int) -> [UInt8] {
        let k = max(2, h / 6)
        var margin = [UInt8](repeating: 0, count: 2 * k)
        var background = [UInt8](repeating: 0, count: w)
        for x in 0..<w {
            for y in 0..<k {
                margin[y] = g[y * w + x]
                margin[k + y] = g[(h - k + y) * w + x]
            }
            margin.sort()
            background[x] = UInt8((Int(margin[k - 1]) + Int(margin[k])) / 2) // np.median, then astype(uint8)
        }
        let smooth = medianFilter(background, window: max(3, min(h | 1, 255)))

        var diff = [Float](repeating: 0, count: w * h)
        var histogram = [Int](repeating: 0, count: 256)
        for i in 0..<(w * h) {
            let d = abs(Float(g[i]) - Float(smooth[i % w]))
            diff[i] = d
            histogram[Int(d)] += 1
        }
        let target = 0.98 * Double(w * h)
        var seen = 0
        var level = 255
        for (value, count) in histogram.enumerated() {
            seen += count
            if Double(seen) >= target {
                level = value
                break
            }
        }
        let scale = 255 / max(Float(level), 24)
        return diff.map { UInt8(255 - min($0 * scale, 255)) }
    }

    /// cv2.medianBlur along one row: odd window, border replicated.
    static func medianFilter(_ values: [UInt8], window: Int) -> [UInt8] {
        let radius = window / 2
        let last = values.count - 1
        var buffer = [UInt8](repeating: 0, count: window)
        return values.indices.map { x in
            for j in 0..<window {
                buffer[j] = values[min(max(x - radius + j, 0), last)]
            }
            buffer.sort()
            return buffer[radius]
        }
    }
}
