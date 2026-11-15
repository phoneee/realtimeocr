import Foundation

/// Port of the model runtime's text clean-up: `fix_scripts(canonical(text))` from
/// runtime/thai_norm.py and runtime/line_engine.py, plus `invalid_share`.
/// Works on Unicode scalars, as Python does, not on Swift Characters
/// (a Thai consonant and its marks are one Character).
public enum ThaiText {
    public static func clean(_ text: String) -> String {
        fixScripts(canonical(text))
    }

    // MARK: canonical

    /// Legacy Thai glyph-variant private-use code points -> standard Thai.
    private static let pua: [UInt32: UInt32] = [
        0xF700: 0x0E10, 0xF701: 0x0E34, 0xF702: 0x0E35, 0xF703: 0x0E36, 0xF704: 0x0E37,
        0xF705: 0x0E48, 0xF706: 0x0E49, 0xF707: 0x0E4A, 0xF708: 0x0E4B, 0xF709: 0x0E4C,
        0xF70A: 0x0E48, 0xF70B: 0x0E49, 0xF70C: 0x0E4A, 0xF70D: 0x0E4B, 0xF70E: 0x0E4C,
        0xF70F: 0x0E0D, 0xF710: 0x0E31, 0xF711: 0x0E4D, 0xF712: 0x0E47, 0xF713: 0x0E48,
        0xF714: 0x0E49, 0xF715: 0x0E4A, 0xF716: 0x0E4B, 0xF717: 0x0E4C, 0xF718: 0x0E38,
        0xF719: 0x0E39, 0xF71A: 0x0E3A,
    ]
    /// Zero-width space/joiners, word joiner, soft hyphen, BOM.
    private static let dropped: Set<UInt32> = [0x200B, 0x200C, 0x200D, 0x2060, 0x00AD, 0xFEFF]
    private static let nbsp: UInt32 = 0x00A0

    private static let vowelMarks = Set("ัิีึืฺุู็".unicodeScalars.map(\.value))
    private static let toneMarks = Set("่้๊๋์๎".unicodeScalars.map(\.value))
    private static let nikhahit: UInt32 = 0x0E4D

    private static let saraAm = try! NSRegularExpression(pattern: "([\\u0E48-\\u0E4B]?)\\u0E4D([\\u0E48-\\u0E4B]?)\\u0E32")
    private static let spaces = try! NSRegularExpression(pattern: "[ \\t\\f\\v]+")

    static func canonical(_ text: String) -> String {
        let translated = String(String.UnicodeScalarView(text.unicodeScalars.compactMap(translate)))
        var out = orderMarks(translated.precomposedStringWithCanonicalMapping)
        out = replace(saraAm, in: out, with: "$1$2\u{0E33}")
        out = out.replacingOccurrences(of: "\u{0E40}\u{0E40}", with: "\u{0E41}", options: .literal)
            .replacingOccurrences(of: "\u{0E33}\u{0E33}", with: "\u{0E33}", options: .literal)
            .replacingOccurrences(of: "\r\n", with: "\n", options: .literal)
            .replacingOccurrences(of: "\r", with: "\n", options: .literal)
        return out.components(separatedBy: "\n")
            .map { replace(spaces, in: $0, with: " ").trimmingCharacters(in: .whitespacesAndNewlines) }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func translate(_ scalar: Unicode.Scalar) -> Unicode.Scalar? {
        let value = scalar.value
        if dropped.contains(value) { return nil }
        if value == nbsp { return " " }
        return pua[value].flatMap(Unicode.Scalar.init) ?? scalar
    }

    private static func markRank(_ value: UInt32) -> Int? {
        if vowelMarks.contains(value) { return 0 }
        if toneMarks.contains(value) { return 1 }
        if value == nikhahit { return 2 }
        return nil
    }

    /// Runs of 2+ marks: vowel, then tone, then nikhahit (stable), a doubled mark kept once.
    private static func orderMarks(_ text: String) -> String {
        var result = String.UnicodeScalarView()
        var run: [Unicode.Scalar] = []
        func flush() {
            if run.count < 2 {
                result.append(contentsOf: run)
            } else {
                let sorted = run.enumerated()
                    .sorted { (markRank($0.element.value)!, $0.offset) < (markRank($1.element.value)!, $1.offset) }
                    .map(\.element)
                // result.last before the run is never a mark (runs are maximal).
                for scalar in sorted where result.last != scalar {
                    result.append(scalar)
                }
            }
            run.removeAll()
        }
        for scalar in text.unicodeScalars {
            if markRank(scalar.value) != nil {
                run.append(scalar)
            } else {
                flush()
                result.append(scalar)
            }
        }
        flush()
        return String(result)
    }

    // MARK: fix_scripts

    /// A lone Latin W glued to Thai letters is พ; a Latin O/0 next to Thai digits is ๐.
    private static let loneW = try! NSRegularExpression(pattern: "(?<![A-Za-z])[Ww](?=\\.?[\\u0E01-\\u0E2E\\u0E40-\\u0E44])")
    private static let zeroInThai = try! NSRegularExpression(pattern: "(?<=[\\u0E50-\\u0E59])[Oo0]|[Oo0](?=[\\u0E50-\\u0E59])")

    static func fixScripts(_ text: String) -> String {
        replace(zeroInThai, in: replace(loneW, in: text, with: "\u{0E1E}"), with: "\u{0E50}")
    }

    private static func replace(_ regex: NSRegularExpression, in text: String, with template: String) -> String {
        regex.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: template)
    }

    // MARK: invalid_share

    private static let spellingMarks = Set("\u{0E31}\u{0E34}\u{0E35}\u{0E36}\u{0E37}\u{0E38}\u{0E39}\u{0E3A}\u{0E47}\u{0E48}\u{0E49}\u{0E4A}\u{0E4B}\u{0E4C}\u{0E4D}\u{0E4E}".unicodeScalars.map(\.value))
    private static let leadingVowels: ClosedRange<UInt32> = 0x0E40...0x0E44
    private static let consonants: ClosedRange<UInt32> = 0x0E01...0x0E2E
    private static let thaiBlock: ClosedRange<UInt32> = 0x0E00...0x0E7F

    /// Share of Thai characters that break Thai spelling: a mark with no consonant
    /// under it, or a leading vowel not followed by a consonant.
    static func invalidShare(_ text: String) -> Double {
        let s = text.unicodeScalars.map(\.value)
        var thai = 0
        var bad = 0
        for (i, ch) in s.enumerated() where thaiBlock.contains(ch) {
            thai += 1
            let prev: UInt32 = i > 0 ? s[i - 1] : 0x20
            if spellingMarks.contains(ch) {
                let onConsonant = consonants.contains(prev)
                let onStackedMark = spellingMarks.contains(prev) && i >= 2 && consonants.contains(s[i - 2])
                if !onConsonant && !onStackedMark { bad += 1 }
            } else if leadingVowels.contains(ch) {
                if !(i + 1 < s.count && consonants.contains(s[i + 1])) { bad += 1 }
            }
        }
        return thai > 0 ? Double(bad) / Double(thai) : 0
    }
}
