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

        let lines: [Line]
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

    func testCleanMatchesReferenceCanonicalAndScriptFixes() throws {
        for item in try expected().canonical {
            XCTAssertEqual(ThaiText.clean(item.input), item.output, item.input.debugDescription)
        }
    }

    func testRejectsLowConfidenceShortAndBrokenThai() {
        XCTAssertTrue(LineReading.isAcceptable(text: "ราคา", confidence: 0.95))
        XCTAssertFalse(LineReading.isAcceptable(text: "ราคา", confidence: 0.79), "under drop_below")
        XCTAssertFalse(LineReading.isAcceptable(text: "ก", confidence: 0.99), "under min_chars")
        XCTAssertFalse(LineReading.isAcceptable(text: "ูี้ซ์ ริ ร้ร้ ไ้", confidence: 0.85), "junk Thai spelling")
    }
}
