import CoreGraphics
import Foundation
import OnnxRuntimeBindings

public struct LineReading: Sendable {
    public let text: String
    public let confidence: Float
    public let prepMs: Double
    public let inferMs: Double
    public let decodeMs: Double

    public var isAcceptable: Bool { Self.isAcceptable(text: text, confidence: confidence) }

    // The model's config.json "rec" line filter.
    static let dropBelow: Float = 0.8
    static let junkBelow: Float = 0.9
    static let junkInvalid = 0.05
    static let minChars = 2

    static func isAcceptable(text: String, confidence: Float) -> Bool {
        guard !text.isEmpty, confidence >= dropBelow else { return false }
        guard text.unicodeScalars.filter({ $0 != " " }).count >= minChars else { return false }
        return !(confidence < junkBelow && ThaiText.invalidShare(text) > junkInvalid)
    }
}

/// One text line in, text out: rec.onnx on ONNX Runtime (CPU), greedy CTC over rec_dict.txt.
/// Not thread-safe: call from one queue.
public final class LineRecognizer {
    public enum Failure: Error {
        case missingResource(String)
        case missingOutput
    }

    private let env: ORTEnv
    private let session: ORTSession
    /// CTC classes: 0 = blank, then the dictionary, then the space PaddleOCR appends.
    private let classes: [String]

    // ponytail: plain CPU EP, 2 threads (A16 has 2 performance cores). Compare XNNPACK / Core ML EPs once on-device timings are in.
    public convenience init(threads: Int32 = 2) throws {
        try self.init(env: ORTEnv(loggingLevel: .warning), threads: threads)
    }

    init(env: ORTEnv, threads: Int32) throws {
        let entries = try String(contentsOf: ModelFile.url("rec_dict", "txt"), encoding: .utf8).unicodeScalars
            .split(separator: "\n")
            .map { String(String.UnicodeScalarView($0)) }
        classes = [""] + entries + [" "]

        self.env = env
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(threads)
        try options.setGraphOptimizationLevel(.all)
        session = try ORTSession(env: env, modelPath: ModelFile.url("rec", "onnx").path, sessionOptions: options)
    }

    public func read(_ image: CGImage) throws -> LineReading {
        try read(RGBAImage(image))
    }

    public func read(_ image: RGBAImage) throws -> LineReading {
        let start = DispatchTime.now()
        let tensor = LinePreprocess.tensor(from: image)
        let prepared = DispatchTime.now()

        let data = tensor.values.withUnsafeBufferPointer {
            NSMutableData(bytes: $0.baseAddress, length: $0.count * MemoryLayout<Float>.stride)
        }
        let shape: [NSNumber] = [1, 3, NSNumber(value: LinePreprocess.height), NSNumber(value: tensor.width)]
        let input = try ORTValue(tensorData: data, elementType: .float, shape: shape)
        let outputs = try session.run(withInputs: ["x": input], outputNames: ["probs"], runOptions: nil)
        guard let probs = outputs["probs"] else { throw Failure.missingOutput }
        let dims = try probs.tensorTypeAndShapeInfo().shape.map(\.intValue) // [1, frames, classes]
        let raw = try probs.tensorData() as Data
        let inferred = DispatchTime.now()

        let (text, confidence) = decode(raw, frames: dims[1], classCount: dims[2])
        let decoded = DispatchTime.now()
        return LineReading(
            text: text, confidence: confidence,
            prepMs: milliseconds(start, prepared), inferMs: milliseconds(prepared, inferred), decodeMs: milliseconds(inferred, decoded)
        )
    }

    /// Greedy CTC: argmax per frame, repeats collapsed, blanks dropped; confidence is the
    /// mean of each kept character's first-frame probability (as `_ctc_detail`).
    private func decode(_ data: Data, frames: Int, classCount: Int) -> (String, Float) {
        data.withUnsafeBytes { raw -> (String, Float) in
            let p = raw.bindMemory(to: Float.self)
            var text = ""
            var kept: [Float] = []
            var previous = -1
            for t in 0..<frames {
                let row = t * classCount
                var best = 0
                for c in 1..<classCount where p[row + c] > p[row + best] {
                    best = c
                }
                if best != previous && best != 0 {
                    text += best < classes.count ? classes[best] : ""
                    kept.append(p[row + best])
                }
                previous = best
            }
            let confidence = kept.isEmpty ? 0 : kept.reduce(0, +) / Float(kept.count)
            return (ThaiText.clean(text), confidence)
        }
    }

    private func milliseconds(_ from: DispatchTime, _ to: DispatchTime) -> Double {
        Double(to.uptimeNanoseconds - from.uptimeNanoseconds) / 1_000_000
    }
}
