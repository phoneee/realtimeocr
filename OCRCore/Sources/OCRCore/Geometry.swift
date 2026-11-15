import CoreGraphics
import Foundation

/// A text-line box: top-left, top-right, bottom-right, bottom-left (the runtime's order_points),
/// in image pixels.
public typealias Quad = [CGPoint]

/// cv2.minAreaRect's result: a rectangle at any angle.
struct RotatedRect {
    let center: CGPoint
    let axis: CGPoint // unit vector along `width`
    let width: Double
    let height: Double

    var side: Double { min(width, height) }

    /// Corners in the runtime's order (cv2.boxPoints + order_points).
    var corners: Quad {
        let normal = CGPoint(x: -axis.y, y: axis.x)
        let u = width / 2
        let v = height / 2
        let signs: [(Double, Double)] = [(-u, -v), (u, -v), (u, v), (-u, v)]
        let points = signs.map { a, b in
            CGPoint(x: center.x + a * axis.x + b * normal.x, y: center.y + a * axis.y + b * normal.y)
        }
        return Geometry.ordered(points)
    }

    /// `_unclip`'s offset distance: area * ratio / perimeter.
    func unclipDistance(ratio: Double) -> Double? {
        let perimeter = 2 * (width + height)
        return perimeter > 0 ? width * height * ratio / perimeter : nil
    }

    /// A rectangle's round-joined offset by `d` (pyclipper) has this as its minAreaRect.
    func grown(by d: Double) -> RotatedRect {
        RotatedRect(center: center, axis: axis, width: width + 2 * d, height: height + 2 * d)
    }
}

enum Geometry {
    static func distance(_ a: CGPoint, _ b: CGPoint) -> Double {
        hypot(a.x - b.x, a.y - b.y)
    }

    /// order_points: sort by x; the left pair and the right pair each by y.
    static func ordered(_ points: [CGPoint]) -> Quad {
        let byX = points.sorted { $0.x < $1.x }
        let left = byX[0..<2].sorted { $0.y < $1.y }
        let right = byX[2..<4].sorted { $0.y < $1.y }
        return [left[0], right[0], right[1], left[1]]
    }

    /// Andrew's monotone chain, counter-clockwise, no collinear points.
    static func convexHull(_ points: [CGPoint]) -> [CGPoint] {
        let sorted = points.sorted { ($0.x, $0.y) < ($1.x, $1.y) }
        guard sorted.count > 2 else { return sorted }
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> Double {
            (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        }
        var lower: [CGPoint] = []
        for p in sorted {
            while lower.count >= 2 && cross(lower[lower.count - 2], lower[lower.count - 1], p) <= 0 { lower.removeLast() }
            lower.append(p)
        }
        var upper: [CGPoint] = []
        for p in sorted.reversed() {
            while upper.count >= 2 && cross(upper[upper.count - 2], upper[upper.count - 1], p) <= 0 { upper.removeLast() }
            upper.append(p)
        }
        return Array(lower.dropLast() + upper.dropLast())
    }

    /// Smallest-area enclosing rectangle: one side lies on a hull edge (rotating calipers).
    static func minAreaRect(_ points: [CGPoint]) -> RotatedRect {
        let hull = convexHull(points)
        guard hull.count > 1 else {
            return RotatedRect(center: hull.first ?? .zero, axis: CGPoint(x: 1, y: 0), width: 0, height: 0)
        }
        var best: RotatedRect?
        var bestArea = Double.infinity
        for i in hull.indices {
            let a = hull[i]
            let b = hull[(i + 1) % hull.count]
            let length = distance(a, b)
            guard length > 0 else { continue }
            let axis = CGPoint(x: (b.x - a.x) / length, y: (b.y - a.y) / length)
            var (minU, maxU, minV, maxV) = (Double.infinity, -Double.infinity, Double.infinity, -Double.infinity)
            for p in hull {
                let u = (p.x - a.x) * axis.x + (p.y - a.y) * axis.y
                let v = -(p.x - a.x) * axis.y + (p.y - a.y) * axis.x
                (minU, maxU, minV, maxV) = (min(minU, u), max(maxU, u), min(minV, v), max(maxV, v))
            }
            let area = (maxU - minU) * (maxV - minV)
            if area < bestArea {
                bestArea = area
                let cu = (minU + maxU) / 2
                let cv = (minV + maxV) / 2
                let center = CGPoint(x: a.x + cu * axis.x - cv * axis.y, y: a.y + cu * axis.y + cv * axis.x)
                best = RotatedRect(center: center, axis: axis, width: maxU - minU, height: maxV - minV)
            }
        }
        // Every edge zero-length: a 1-pixel region (its leftmost and rightmost pixel coincide).
        return best ?? RotatedRect(center: hull[0], axis: CGPoint(x: 1, y: 0), width: 0, height: 0)
    }

    /// Point in a convex quad, edges included.
    static func contains(_ quad: Quad, _ p: CGPoint) -> Bool {
        var positive = false
        var negative = false
        for i in 0..<4 {
            let a = quad[i]
            let b = quad[(i + 1) % 4]
            let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
            if cross > 0 { positive = true }
            if cross < 0 { negative = true }
        }
        return !(positive && negative)
    }

    /// crop_line: the quad warped to an upright w x h image, bicubic (cv2.INTER_CUBIC), border replicated.
    static func crop(_ image: RGBAImage, _ quad: Quad) -> RGBAImage {
        let w = max(1, Int(max(distance(quad[0], quad[1]), distance(quad[2], quad[3]))))
        let h = max(1, Int(max(distance(quad[0], quad[3]), distance(quad[1], quad[2]))))
        let map = squareToQuad(quad)
        var out = [UInt8](repeating: 0, count: w * h * 4)
        image.pixels.withUnsafeBufferPointer { src in
            for y in 0..<h {
                for x in 0..<w {
                    let p = map(Double(x) / Double(w), Double(y) / Double(h))
                    sampleBicubic(src, image.width, image.height, p.x, p.y, into: &out, at: (y * w + x) * 4)
                }
            }
        }
        return try! RGBAImage(width: w, height: h, pixels: out) // sizes match by construction
    }

    /// Projective map of the unit square onto the quad, corner for corner (Heckbert).
    static func squareToQuad(_ q: Quad) -> (Double, Double) -> CGPoint {
        let (x0, y0, x1, y1, x2, y2, x3, y3) = (q[0].x, q[0].y, q[1].x, q[1].y, q[2].x, q[2].y, q[3].x, q[3].y)
        let sx = x0 - x1 + x2 - x3
        let sy = y0 - y1 + y2 - y3
        let dx1 = x1 - x2, dx2 = x3 - x2, dy1 = y1 - y2, dy2 = y3 - y2
        let den = dx1 * dy2 - dx2 * dy1
        let g = (sx == 0 && sy == 0) || den == 0 ? 0 : (sx * dy2 - dx2 * sy) / den
        let hh = (sx == 0 && sy == 0) || den == 0 ? 0 : (dx1 * sy - sx * dy1) / den
        let a = x1 - x0 + g * x1, b = x3 - x0 + hh * x3, c = x0
        let d = y1 - y0 + g * y1, e = y3 - y0 + hh * y3, f = y0
        return { u, v in
            let z = g * u + hh * v + 1
            return CGPoint(x: (a * u + b * v + c) / z, y: (d * u + e * v + f) / z)
        }
    }

    /// cv2's cubic kernel, A = -0.75: weights for the 4 taps around a fractional offset t.
    private static func cubicWeights(_ t: Double) -> (Double, Double, Double, Double) {
        let a = -0.75
        let w0 = ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a
        let w1 = ((a + 2) * t - (a + 3)) * t * t + 1
        let w2 = ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1
        return (w0, w1, w2, 1 - w0 - w1 - w2)
    }

    private static func sampleBicubic(
        _ src: UnsafeBufferPointer<UInt8>, _ w: Int, _ h: Int, _ x: Double, _ y: Double,
        into out: inout [UInt8], at offset: Int
    ) {
        guard x.isFinite, y.isFinite else { return } // degenerate quad: leave the pixel black
        let fx = x.rounded(.down), fy = y.rounded(.down)
        let (wx0, wx1, wx2, wx3) = cubicWeights(x - fx)
        let (wy0, wy1, wy2, wy3) = cubicWeights(y - fy)
        let wx = [wx0, wx1, wx2, wx3], wy = [wy0, wy1, wy2, wy3]
        let xs = (-1...2).map { min(max(Int(fx) + $0, 0), w - 1) }
        let ys = (-1...2).map { min(max(Int(fy) + $0, 0), h - 1) }
        for c in 0..<4 {
            var value = 0.0
            for j in 0..<4 {
                var row = 0.0
                for i in 0..<4 { row += Double(src[(ys[j] * w + xs[i]) * 4 + c]) * wx[i] }
                value += row * wy[j]
            }
            out[offset + c] = UInt8(clamping: Int(value.rounded()))
        }
    }
}
