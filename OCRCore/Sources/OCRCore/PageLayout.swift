import CoreGraphics
import Foundation

/// Reading order and fragment filtering, as runtime/line_engine.py.
// ponytail: one block per page. The runtime also splits prose columns (_blocks, an XY-cut);
// add it when multi-column pages show up in testing.
enum PageLayout {
    struct Extent {
        let x0: Double, x1: Double, y0: Double, y1: Double

        init(_ points: [CGPoint]) {
            let xs = points.map { Double($0.x) }
            let ys = points.map { Double($0.y) }
            (x0, x1, y0, y1) = (xs.min()!, xs.max()!, ys.min()!, ys.max()!)
        }
    }

    static func median(_ values: [Double]) -> Double {
        let s = values.sorted()
        guard !s.isEmpty else { return 0 }
        return s.count % 2 == 1 ? s[s.count / 2] : (s[s.count / 2 - 1] + s[s.count / 2]) / 2
    }

    /// line_height: median height of line-shaped boxes (at least twice as wide as tall).
    static func lineHeight(_ quads: [Quad]) -> Double {
        guard !quads.isEmpty else { return 1 }
        let heights = quads.map { max(Geometry.distance($0[0], $0[3]), 1) }
        let lines = zip(quads, heights).filter { Geometry.distance($0.0[0], $0.0[1]) >= 2 * $0.1 }.map(\.1)
        return median(lines.count >= 3 ? lines : heights)
    }

    /// page_skew: median angle in degrees of boxes at least 3x wider than tall.
    static func skew(_ quads: [Quad]) -> Double {
        let angles = quads.compactMap { q -> Double? in
            let dx = q[1].x - q[0].x, dy = q[1].y - q[0].y
            guard hypot(dx, dy) >= 3 * max(Geometry.distance(q[0], q[3]), 1) else { return nil }
            return atan2(dy, dx) * 180 / .pi
        }
        return median(angles)
    }

    /// reading_order without column blocks: rows top to bottom by vertical overlap in deskewed
    /// coordinates, each left to right. Small boxes join the nearest row without moving it.
    static func rows(_ quads: [Quad]) -> [[Int]] {
        guard !quads.isEmpty else { return [] }
        let a = -skew(quads) * .pi / 180
        let (c, s) = (cos(a), sin(a))
        let extents = quads.map { q in Extent(q.map { CGPoint(x: $0.x * c - $0.y * s, y: $0.x * s + $0.y * c) }) }
        let hmed = lineHeight(quads)
        let cy = extents.map { ($0.y0 + $0.y1) / 2 }
        let rx = extents.map { ($0.x0 + $0.x1) / 2 }
        let hs = quads.map { max(Geometry.distance($0[0], $0[3]), 1) }
        let tol = 0.5 * hmed

        let tall = quads.indices.filter { hs[$0] >= 0.6 * hmed }
        let full = tall.isEmpty ? Array(quads.indices) : tall
        let small = quads.indices.filter { !Set(full).contains($0) }
        var rows: [[Int]] = []
        for i in full.sorted(by: { cy[$0] < cy[$1] }) {
            if let last = rows.last, abs(cy[i] - last.map { cy[$0] }.reduce(0, +) / Double(last.count)) <= tol {
                rows[rows.count - 1].append(i)
            } else {
                rows.append([i])
            }
        }
        var centres = rows.map { r in r.map { cy[$0] }.reduce(0, +) / Double(r.count) }
        for i in small {
            let d = centres.map { abs(cy[i] - $0) }
            if let nearest = d.indices.min(by: { d[$0] < d[$1] }), d[nearest] <= tol + 0.5 * hs[i] {
                rows[nearest].append(i)
            } else {
                rows.append([i])
                centres.append(cy[i])
            }
        }
        return rows.indices.sorted { centres[$0] < centres[$1] }.map { r in rows[r].sorted { rx[$0] < rx[$1] } }
    }

    /// _fragments: boxes that are pieces of a neighbouring line (a stacked mark or a tail
    /// detected alone): at most 3 characters, under 35 % of the line height, touching a
    /// full-height line above or below.
    static func fragments(_ quads: [Quad], texts: [String]) -> Set<Int> {
        guard quads.count >= 3 else { return [] }
        let hs = quads.map { Geometry.distance($0[0], $0[3]) }
        let hmed = lineHeight(quads)
        let ext = quads.map(Extent.init)
        var out = Set<Int>()
        for i in quads.indices {
            guard hs[i] < 0.35 * hmed, texts[i].unicodeScalars.filter({ $0 != " " }).count <= 3 else { continue }
            let b = ext[i]
            let touches = quads.indices.contains { j in
                guard j != i, hs[j] >= 0.6 * hmed else { return false }
                let c = ext[j]
                return c.x0 <= b.x1 && b.x0 <= c.x1 && min(abs(b.y0 - c.y1), abs(c.y0 - b.y1)) <= 0.5 * hmed
            }
            if touches { out.insert(i) }
        }
        return out
    }
}
