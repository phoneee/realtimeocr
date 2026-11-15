import CoreGraphics
import Foundation
import OnnxRuntimeBindings

/// det.onnx (PP-OCRv6-tiny) + DB post-processing, as `LineOCR.detect`.
final class TextDetector {
    // det.yml PostProcess values, which override the runtime defaults.
    static let thresh: Float = 0.2
    static let boxThresh: Float = 0.4
    static let unclipRatio = 1.4
    static let minSize = 3.0
    static let maxCandidates = 3000
    // ImageNet statistics in the runtime's channel order: it feeds BGR.
    private static let mean: [Float] = [0.485, 0.456, 0.406]
    private static let std: [Float] = [0.229, 0.224, 0.225]

    private let session: ORTSession

    init(env: ORTEnv, threads: Int32) throws {
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(threads)
        try options.setGraphOptimizationLevel(.all)
        session = try ORTSession(env: env, modelPath: try ModelFile.url("det", "onnx").path, sessionOptions: options)
    }

    /// Line boxes in `image` pixels. `limit`: the detector's long side, the speed/small-text trade-off.
    func detect(_ image: RGBAImage, limit: Int) throws -> [Quad] {
        let input = Self.input(image, limit: limit)
        let data = input.values.withUnsafeBufferPointer {
            NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Float>.stride)
        }
        let shape: [NSNumber] = [1, 3, NSNumber(value: input.height), NSNumber(value: input.width)]
        let tensor = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        let outputs = try session.run(withInputs: ["x": tensor], outputNames: ["fetch_name_0"], runOptions: nil)
        guard let output = outputs["fetch_name_0"] else { throw LineRecognizer.Failure.missingOutput }
        let dims = try output.tensorTypeAndShapeInfo().shape.map(\.intValue) // [1, 1, h, w]
        let prob = (try output.tensorData() as Data).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return Self.boxes(
            prob: prob, width: dims[3], height: dims[2],
            scale: CGPoint(x: Double(input.paddedWidth) / Double(dims[3]), y: Double(input.paddedHeight) / Double(dims[2])),
            pad: Double(input.pad), imageWidth: image.width, imageHeight: image.height
        )
    }

    struct Input {
        let values: [Float]
        let width: Int
        let height: Int
        let pad: Int
        let paddedWidth: Int
        let paddedHeight: Int
    }

    /// White border, then the padded image scaled so its long side is `limit` (multiples of 32,
    /// at most 4x), bilinear, normalised, channel-major B, G, R.
    static func input(_ image: RGBAImage, limit: Int) -> Input {
        let pad = max(8, Int(0.05 * Double(min(image.width, image.height))))
        let w = image.width + 2 * pad
        let h = image.height + 2 * pad
        let s = min(Double(limit) / Double(max(w, h)), 4.0)
        let rw = max(32, Int((Double(w) * s / 32).rounded(.toNearestOrEven)) * 32) // Python round()
        let rh = max(32, Int((Double(h) * s / 32).rounded(.toNearestOrEven)) * 32)

        func taps(_ d: Int, scale: Double, size: Int) -> (Int, Int, Double) {
            let f = (Double(d) + 0.5) * scale - 0.5
            var i = Int(f.rounded(.down))
            var t = f - Double(i)
            if i < 0 { i = 0; t = 0 }
            if i >= size - 1 { i = size - 1; t = 0 }
            return (i, min(i + 1, size - 1), t)
        }
        let xTaps = (0..<rw).map { taps($0, scale: Double(w) / Double(rw), size: w) }
        let plane = rw * rh
        var values = [Float](repeating: 0, count: 3 * plane)
        image.pixels.withUnsafeBufferPointer { px in
            func at(_ x: Int, _ y: Int, _ c: Int) -> Double {
                let ix = x - pad, iy = y - pad
                guard ix >= 0, iy >= 0, ix < image.width, iy < image.height else { return 255 }
                return Double(px[(iy * image.width + ix) * 4 + c])
            }
            for y in 0..<rh {
                let (y0, y1, ty) = taps(y, scale: Double(h) / Double(rh), size: h)
                for x in 0..<rw {
                    let (x0, x1, tx) = xTaps[x]
                    for (channel, rgba) in [2, 1, 0].enumerated() { // B, G, R
                        let top = at(x0, y0, rgba) * (1 - tx) + at(x1, y0, rgba) * tx
                        let bottom = at(x0, y1, rgba) * (1 - tx) + at(x1, y1, rgba) * tx
                        let byte = Float((top * (1 - ty) + bottom * ty).rounded())
                        values[channel * plane + y * rw + x] = (byte / 255 - mean[channel]) / std[channel]
                    }
                }
            }
        }
        return Input(values: values, width: rw, height: rh, pad: pad, paddedWidth: w, paddedHeight: h)
    }

    /// DB post-processing: threshold, one box per 8-connected region (its min-area rectangle),
    /// kept if wide enough and confident enough inside, grown by `unclip`, mapped back to the image.
    // ponytail: outer regions only; cv2.findContours(RETR_LIST) also returns hole contours,
    // which on line blobs are rare and score low.
    static func boxes(
        prob: [Float], width: Int, height: Int, scale: CGPoint, pad: Double, imageWidth: Int, imageHeight: Int
    ) -> [Quad] {
        var visited = [Bool](repeating: false, count: width * height)
        var stack: [Int] = []
        var quads: [Quad] = []
        var candidates = 0
        for start in 0..<(width * height) where prob[start] > thresh && !visited[start] {
            guard candidates < maxCandidates else { break }
            candidates += 1
            var rows: [Int: (Int, Int)] = [:] // y -> leftmost, rightmost x
            visited[start] = true
            stack.append(start)
            while let i = stack.popLast() {
                let x = i % width, y = i / width
                let extent = rows[y]
                rows[y] = (min(extent?.0 ?? x, x), max(extent?.1 ?? x, x))
                for dy in -1...1 {
                    for dx in -1...1 {
                        let nx = x + dx, ny = y + dy
                        guard nx >= 0, ny >= 0, nx < width, ny < height else { continue }
                        let n = ny * width + nx
                        if !visited[n] && prob[n] > thresh {
                            visited[n] = true
                            stack.append(n)
                        }
                    }
                }
            }
            let points = rows.flatMap { y, e in [CGPoint(x: e.0, y: y), CGPoint(x: e.1, y: y)] }
            let rect = Geometry.minAreaRect(points)
            guard rect.side >= minSize, score(prob, width, height, rect.corners) >= boxThresh else { continue }
            // ponytail: float geometry. pyclipper truncates the corners to integers first, which moves
            // the runtime's boxes up-left by under a pixel; copying that flipped other characters.
            guard let d = rect.unclipDistance(ratio: unclipRatio) else { continue }
            let grown = rect.grown(by: d)
            guard grown.side >= minSize + 2 else { continue }
            let quad = grown.corners.map {
                CGPoint(
                    x: min(max($0.x * scale.x - pad, 0), Double(imageWidth - 1)),
                    y: min(max($0.y * scale.y - pad, 0), Double(imageHeight - 1))
                )
            }
            let xs = quad.map(\.x), ys = quad.map(\.y)
            if xs.max()! - xs.min()! >= 2 && ys.max()! - ys.min()! >= 2 {
                quads.append(quad)
            }
        }
        return quads
    }

    /// _box_score: mean probability inside the box (corners truncated to whole pixels, as fillPoly).
    static func score(_ prob: [Float], _ width: Int, _ height: Int, _ box: Quad) -> Float {
        let xs = box.map(\.x), ys = box.map(\.y)
        let x0 = min(max(Int(xs.min()!.rounded(.down)), 0), width - 1)
        let x1 = min(max(Int(xs.max()!.rounded(.up)), 0), width - 1)
        let y0 = min(max(Int(ys.min()!.rounded(.down)), 0), height - 1)
        let y1 = min(max(Int(ys.max()!.rounded(.up)), 0), height - 1)
        let poly = box.map { CGPoint(x: Double(Int($0.x - Double(x0))), y: Double(Int($0.y - Double(y0)))) }
        var sum: Float = 0
        var count = 0
        for y in y0...y1 {
            for x in x0...x1 where Geometry.contains(poly, CGPoint(x: x - x0, y: y - y0)) {
                sum += prob[y * width + x]
                count += 1
            }
        }
        return count > 0 ? sum / Float(count) : 0
    }
}
