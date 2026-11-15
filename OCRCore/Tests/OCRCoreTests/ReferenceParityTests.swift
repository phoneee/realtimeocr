import CoreGraphics
import Foundation
import ImageIO
import XCTest
@testable import OCRCore

/// Fixtures come from tools/make_fixtures.py, which runs the model's own Python runtime.
/// The Swift port must read the same crops the same way.
final class ReferenceParityTests: XCTestCase {
    private struct Expected: Decodable {
        struct Line: Decodable {
            let file: String
            let truth: String
            let text: String
            let confidence: Float
        }

        struct Case: Decodable {
            let input: String
            let output: String
        }

        struct Page: Decodable {
            let file: String
            let text: String
            let lines: Int
        }

        let lines: [Line]
        let pages: [Page]
        let canonical: [Case]
    }

    private func fixtureURL(_ name: String) throws -> URL {
        try XCTUnwrap(Bundle.module.url(forResource: name, withExtension: nil, subdirectory: "Fixtures"))
    }

    private func expected() throws -> Expected {
        try JSONDecoder().decode(Expected.self, from: Data(contentsOf: fixtureURL("expected.json")))
    }

    private func image(_ name: String) throws -> CGImage {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(fixtureURL(name) as CFURL, nil))
        return try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    func testReadsLinesLikeReferenceRuntime() throws {
        let recognizer = try LineRecognizer()
        for line in try expected().lines {
            let reading = try recognizer.read(image(line.file))
            XCTAssertEqual(reading.text, line.text, line.file)
            XCTAssertEqual(reading.confidence, line.confidence, accuracy: 0.01, line.file)
        }
    }

    /// Same lines, near-same text. Boxes agree with the runtime's within 1-2 px, but it rounds
    /// through integers (cv2 fixed-point resize, pyclipper), and a line's last glyph can read
    /// differently over 1 px. So: exact line count, and at most 2 % of characters differing.
    func testReadsPagesLikeReferenceRuntime() throws {
        let reader = try PageReader()
        for page in try expected().pages {
            let reading = try reader.read(RGBAImage(image(page.file)), limit: 1280)
            XCTAssertEqual(reading.lines.count, page.lines, page.file)
            let edits = Self.editDistance(Array(reading.text.unicodeScalars), Array(page.text.unicodeScalars))
            XCTAssertLessThanOrEqual(Double(edits), 0.02 * Double(page.text.unicodeScalars.count), "\(page.file):\n\(reading.text)")
        }
    }

    private static func editDistance<T: Equatable>(_ a: [T], _ b: [T]) -> Int {
        var row = Array(0...b.count)
        for i in 1...max(a.count, 1) where !a.isEmpty {
            var previous = row[0]
            row[0] = i
            for j in 1...max(b.count, 1) where !b.isEmpty {
                let current = row[j]
                row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (a[i - 1] == b[j - 1] ? 0 : 1))
                previous = current
            }
        }
        return a.isEmpty ? b.count : row[b.count]
    }

    func testCleanMatchesReferenceCanonicalAndScriptFixes() throws {
        for item in try expected().canonical {
            XCTAssertEqual(ThaiText.clean(item.input), item.output, item.input.debugDescription)
        }
    }

    func testOnePixelRegionGivesEmptyRectInsteadOfCrashing() {
        let speckle = CGPoint(x: 5, y: 7)
        XCTAssertEqual(Geometry.minAreaRect([speckle, speckle]).side, 0) // a 1-pixel region: leftmost = rightmost
    }

    func testRejectsLowConfidenceShortAndBrokenThai() {
        XCTAssertTrue(LineReading.isAcceptable(text: "ราคา", confidence: 0.95))
        XCTAssertFalse(LineReading.isAcceptable(text: "ราคา", confidence: 0.79), "under drop_below")
        XCTAssertFalse(LineReading.isAcceptable(text: "ก", confidence: 0.99), "under min_chars")
        XCTAssertFalse(LineReading.isAcceptable(text: "ูี้ซ์ ริ ร้ร้ ไ้", confidence: 0.85), "junk Thai spelling")
    }
}
