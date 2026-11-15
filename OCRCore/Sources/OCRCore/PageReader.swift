import CoreGraphics
import Foundation
import OnnxRuntimeBindings

public struct PageLine: Sendable {
    public let text: String
    public let confidence: Float
    public let quad: Quad
}

public struct PageReading: Sendable {
    /// Kept lines in reading order.
    public let lines: [PageLine]
    /// Rows joined by newlines, a row's lines by spaces.
    public let text: String
    public let detected: Int
    public let detMs: Double
    public let recMs: Double
}

/// Whole image in, text in reading order out: detect lines, read each, order them,
/// drop fragments and lines the model's filter rejects (LineOCR.read).
// ponytail: no page orientation (sideways / upside-down), deskew-and-redetect, gap splitting or
// tall boxes read turned. A phone held upright does not need them; add when testing says so.
public final class PageReader {
    public static let defaultLimit = 1280

    private let detector: TextDetector
    private let recognizer: LineRecognizer

    public init(threads: Int32 = 2) throws {
        let env = try ORTEnv(loggingLevel: .warning)
        detector = try TextDetector(env: env, threads: threads)
        recognizer = try LineRecognizer(env: env, threads: threads)
    }

    public func read(_ image: RGBAImage, limit: Int = PageReader.defaultLimit) throws -> PageReading {
        let start = DispatchTime.now()
        let quads = try detector.detect(image, limit: limit)
        let detected = DispatchTime.now()
        let readings = try quads.map { try recognizer.read(Geometry.crop(image, $0)) }
        let recognized = DispatchTime.now()

        let fragments = PageLayout.fragments(quads, texts: readings.map(\.text))
        var lines: [PageLine] = []
        var rowTexts: [String] = []
        for row in PageLayout.rows(quads) {
            let kept = row.filter { !fragments.contains($0) && readings[$0].isAcceptable }
            lines += kept.map { PageLine(text: readings[$0].text, confidence: readings[$0].confidence, quad: quads[$0]) }
            if !kept.isEmpty {
                rowTexts.append(kept.map { readings[$0].text }.joined(separator: " "))
            }
        }
        return PageReading(
            lines: lines, text: rowTexts.joined(separator: "\n"), detected: quads.count,
            detMs: Self.milliseconds(start, detected), recMs: Self.milliseconds(detected, recognized)
        )
    }

    static func milliseconds(_ from: DispatchTime, _ to: DispatchTime) -> Double {
        Double(to.uptimeNanoseconds - from.uptimeNanoseconds) / 1_000_000
    }
}
