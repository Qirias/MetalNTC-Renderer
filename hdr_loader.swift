import Foundation

/// Supports the new-style adaptive-RLE scanlines that every modern HDRI uses
/// (Poly Haven, HDRI Haven, ...), with a flat per-scanline fallback. Output is
/// tightly-packed RGBA32F (alpha = 1), row 0 = top, ready for an equirect texture.
enum HDRLoader {
    struct Image { var width: Int; var height: Int; var rgba: [Float] }

    enum Error: Swift.Error, CustomStringConvertible {
        case badFile
        var description: String { "HDR: not a supported Radiance/RGBE file" }
    }

    static func load(url: URL) throws -> Image {
        let bytes = [UInt8](try Data(contentsOf: url))
        var cursor = 0

        func readLine() -> String {
            let start = cursor
            while cursor < bytes.count, bytes[cursor] != 0x0A { cursor += 1 }
            let line = String(decoding: bytes[start..<cursor], as: UTF8.self)
            cursor += 1                                  // skip '\n'
            return line
        }

        guard readLine().hasPrefix("#?") else { throw Error.badFile }   // magic
        while !readLine().isEmpty && cursor < bytes.count {}            // skip header vars

        // Resolution line: "-Y <height> +X <width>".
        let tokens = readLine().split(separator: " ")
        guard tokens.count == 4, tokens[0] == "-Y", tokens[2] == "+X",
              let height = Int(tokens[1]), let width = Int(tokens[3]) else { throw Error.badFile }

        var rgba = [Float](repeating: 0, count: width * height * 4)

        // Shared-exponent RGBE -> linear float, written to rgba[offset ..< offset+4].
        func storePixel(_ offset: Int, _ red: UInt8, _ green: UInt8, _ blue: UInt8, _ exponent: UInt8) {
            rgba[offset + 3] = 1
            guard exponent != 0 else { return }
            let scale = Float(exp2(Double(Int(exponent) - 136)))   // ldexp(1, exponent-128-8)
            rgba[offset]     = Float(red)   * scale
            rgba[offset + 1] = Float(green) * scale
            rgba[offset + 2] = Float(blue)  * scale
        }

        var scanline = [UInt8](repeating: 0, count: width * 4)   // one row, planar RGBE

        for row in 0..<height {
            guard cursor + 4 <= bytes.count else { throw Error.badFile }
            let isAdaptiveRLE = bytes[cursor] == 2 && bytes[cursor + 1] == 2
                             && (Int(bytes[cursor + 2]) << 8 | Int(bytes[cursor + 3])) == width

            if isAdaptiveRLE {
                cursor += 4
                for channel in 0..<4 {                            // 4 RLE-encoded channel planes
                    var column = 0
                    while column < width {
                        let count = Int(bytes[cursor]); cursor += 1
                        if count > 128 {                          // run: (count - 128) copies
                            let value = bytes[cursor]; cursor += 1
                            for _ in 0..<(count - 128) { scanline[column * 4 + channel] = value; column += 1 }
                        } else {                                  // literal: `count` raw bytes
                            for _ in 0..<count { scanline[column * 4 + channel] = bytes[cursor]; cursor += 1; column += 1 }
                        }
                    }
                }
                for column in 0..<width {
                    storePixel((row * width + column) * 4,
                               scanline[column * 4], scanline[column * 4 + 1],
                               scanline[column * 4 + 2], scanline[column * 4 + 3])
                }
            } else {                                              // flat: `width` raw RGBE pixels
                for column in 0..<width {
                    storePixel((row * width + column) * 4,
                               bytes[cursor], bytes[cursor + 1], bytes[cursor + 2], bytes[cursor + 3])
                    cursor += 4
                }
            }
        }

        return Image(width: width, height: height, rgba: rgba)
    }
}
