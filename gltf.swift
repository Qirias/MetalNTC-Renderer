import Foundation
import simd

/// Reads what the renderer draws: every mesh primitive's positions, normals,
/// uvs and indices, each with its world transform (accumulated down the node
/// hierarchy) and glTF material name. It ignores animations, skins, cameras.
/// Material textures are not read here -- those come from the .ntc payload,
/// whose channel layout is described by the manifest.json beside the .gltf.
struct GLTF {

    /// One drawable primitive with its world-space transform (accumulated down
    /// the node hierarchy) and its glTF material name; the renderer loads
    /// `<materialName>.ntc` beside the glTF.
    struct Submesh {
        var positions: [SIMD3<Float>]
        var normals:   [SIMD3<Float>]
        var uvs:       [SIMD2<Float>]
        /// glTF TANGENT is a VEC4: xyz is the tangent, w is the handedness sign
        /// (+1/-1) for reconstructing the bitangent. Empty when the primitive
        /// has no TANGENT -- the shader then falls back to the geometric normal.
        var tangents:  [SIMD4<Float>]
        var indices:   [UInt32]
        var transform: simd_float4x4
        var materialName: String
    }

    struct Scene {
        var submeshes: [Submesh]
    }

    enum Error: Swift.Error, CustomStringConvertible {
        case notJSON
        case missing(String)
        case unsupported(String)

        var description: String {
            switch self {
            case .notJSON:            return "glTF: file is not valid JSON"
            case .missing(let what):  return "glTF: missing \(what)"
            case .unsupported(let w): return "glTF: unsupported \(w)"
            }
        }
    }

    // MARK: JSON shapes

    private struct Doc: Decodable {
        struct Buffer:     Decodable { let uri: String?; let byteLength: Int }
        struct BufferView: Decodable {
            let buffer: Int
            let byteOffset: Int?
            let byteLength: Int
            let byteStride: Int?
        }
        struct Accessor: Decodable {
            let bufferView: Int?
            let byteOffset: Int?
            let componentType: Int
            let count: Int
            let type: String
        }
        struct Primitive: Decodable {
            let attributes: [String: Int]
            let indices: Int?
            let material: Int?
        }
        struct Mesh: Decodable { let primitives: [Primitive] }
        struct Node: Decodable {
            let mesh: Int?
            let matrix: [Float]?
            let translation: [Float]?
            let rotation: [Float]?
            let scale: [Float]?
            let children: [Int]?
        }
        struct Material: Decodable { let name: String? }
        struct SceneDef: Decodable { let nodes: [Int]? }

        let buffers: [Buffer]
        let bufferViews: [BufferView]
        let accessors: [Accessor]
        let meshes: [Mesh]
        let nodes: [Node]
        let materials: [Material]?
        let scenes: [SceneDef]?
        let scene: Int?
    }

    // glTF componentType enum
    private static let UNSIGNED_SHORT = 5123
    private static let UNSIGNED_INT   = 5125
    private static let FLOAT          = 5126

    static func loadScene(at url: URL) throws -> Scene {
        let json = try Data(contentsOf: url)
        guard let doc = try? JSONDecoder().decode(Doc.self, from: json) else {
            throw Error.notJSON
        }
        let baseDir = url.deletingLastPathComponent()
        let bufferData = try loadBufferData(doc, baseDir)

        // Depth-first from the scene roots, accumulating each node's transform.
        // Bad indices trap on the array access; that is the "crash if malformed".
        var submeshes: [Submesh] = []
        func visit(_ nodeIdx: Int, _ parent: simd_float4x4) throws {
            let node  = doc.nodes[nodeIdx]
            let world = parent * transform(of: node)
            if let meshIdx = node.mesh {
                for prim in doc.meshes[meshIdx].primitives {
                    let geom = try readPrimitive(doc, bufferData, prim)
                    submeshes.append(Submesh(positions: geom.positions,
                                             normals:   geom.normals,
                                             uvs:       geom.uvs,
                                             tangents:  geom.tangents,
                                             indices:   geom.indices,
                                             transform: world,
                                             materialName: doc.materials![prim.material!].name!))
                }
            }
            for child in node.children ?? [] { try visit(child, world) }
        }
        for root in doc.scenes![doc.scene ?? 0].nodes ?? [] {
            try visit(root, matrix_identity_float4x4)
        }
        return Scene(submeshes: submeshes)
    }

    // Resolve external .bin buffers. GLB and base64 data: URIs are rejected
    // rather than silently mis-parsed.
    private static func loadBufferData(_ doc: Doc, _ baseDir: URL) throws -> [Data] {
        var bufferData: [Data] = []
        for buffer in doc.buffers {
            guard let uri = buffer.uri else { throw Error.unsupported("GLB / embedded buffers") }
            if uri.hasPrefix("data:") { throw Error.unsupported("base64 data: URIs") }
            let decoded = uri.removingPercentEncoding ?? uri
            bufferData.append(try Data(contentsOf: baseDir.appendingPathComponent(decoded)))
        }
        return bufferData
    }

    private static func readPrimitive(_ doc: Doc, _ bufferData: [Data], _ prim: Doc.Primitive)
        throws -> (positions: [SIMD3<Float>], normals: [SIMD3<Float>], uvs: [SIMD2<Float>],
                   tangents: [SIMD4<Float>], indices: [UInt32]) {
        guard let posIdx = prim.attributes["POSITION"] else { throw Error.missing("POSITION attribute") }
        guard let uvIdx  = prim.attributes["TEXCOORD_0"] else { throw Error.missing("TEXCOORD_0 attribute") }

        let positions: [SIMD3<Float>] = try readVec3(doc, bufferData, posIdx)
        let uvs:       [SIMD2<Float>] = try readVec2(doc, bufferData, uvIdx)

        let normals: [SIMD3<Float>]
        if let nrmIdx = prim.attributes["NORMAL"] {
            normals = try readVec3(doc, bufferData, nrmIdx)
        } else {
            normals = [SIMD3<Float>](repeating: SIMD3<Float>(0, 1, 0), count: positions.count)
        }

        // TANGENT is optional; left empty when absent so the shader can fall back.
        let tangents: [SIMD4<Float>]
        if let tanIdx = prim.attributes["TANGENT"] {
            tangents = try readVec4(doc, bufferData, tanIdx)
        } else {
            tangents = []
        }

        let indices: [UInt32]
        if let idxAccessor = prim.indices {
            indices = try readIndices(doc, bufferData, idxAccessor)
        } else {
            indices = (0..<UInt32(positions.count)).map { $0 }
        }
        return (positions, normals, uvs, tangents, indices)
    }

    // MARK: node transform

    private static func transform(of node: Doc.Node) -> simd_float4x4 {
        // A node has either `matrix` (column-major, matching simd) or T/R/S.
        if let m = node.matrix {
            return simd_float4x4(SIMD4<Float>(m[0],  m[1],  m[2],  m[3]),
                                 SIMD4<Float>(m[4],  m[5],  m[6],  m[7]),
                                 SIMD4<Float>(m[8],  m[9],  m[10], m[11]),
                                 SIMD4<Float>(m[12], m[13], m[14], m[15]))
        }

        // Compose T * R * S, per the glTF spec's stated order.
        var out = matrix_identity_float4x4
        if let t = node.translation {
            out.columns.3 = SIMD4<Float>(t[0], t[1], t[2], 1)
        }
        if let r = node.rotation {
            // glTF quaternions are (x, y, z, w), matching simd_quatf's storage.
            out = out * simd_float4x4(simd_quatf(ix: r[0], iy: r[1], iz: r[2], r: r[3]))
        }
        if let s = node.scale {
            var m = matrix_identity_float4x4
            m.columns.0.x = s[0]
            m.columns.1.y = s[1]
            m.columns.2.z = s[2]
            out = out * m
        }
        return out
    }

    // MARK: accessor readers

    /// Returns the bytes an accessor covers, along with the stride between
    /// consecutive elements. glTF allows interleaved vertex data via
    /// `byteStride`, so never assume elements are tightly packed.
    private static func accessorBytes(_ doc: Doc,
                                      _ bufferData: [Data],
                                      _ index: Int,
                                      elementSize: Int) throws -> (Data, offset: Int, stride: Int, count: Int) {
        guard index < doc.accessors.count else { throw Error.missing("accessor \(index)") }
        let acc = doc.accessors[index]
        guard let viewIdx = acc.bufferView, viewIdx < doc.bufferViews.count else {
            throw Error.unsupported("accessor without a bufferView (sparse?)")
        }
        let view = doc.bufferViews[viewIdx]
        guard view.buffer < bufferData.count else { throw Error.missing("buffer \(view.buffer)") }

        let offset = (view.byteOffset ?? 0) + (acc.byteOffset ?? 0)
        let stride = view.byteStride ?? elementSize
        let needed = offset + (acc.count - 1) * stride + elementSize
        let bytes = bufferData[view.buffer]
        guard needed <= bytes.count else {
            throw Error.missing("accessor \(index) reads past end of buffer")
        }
        return (bytes, offset, stride, acc.count)
    }

    private static func readVec3(_ doc: Doc, _ bufferData: [Data], _ index: Int) throws -> [SIMD3<Float>] {
        guard doc.accessors[index].componentType == FLOAT,
              doc.accessors[index].type == "VEC3" else {
            throw Error.unsupported("VEC3 accessor that is not float")
        }
        let (bytes, offset, stride, count) = try accessorBytes(doc, bufferData, index, elementSize: 12)
        var out = [SIMD3<Float>](repeating: .zero, count: count)
        bytes.withUnsafeBytes { raw in
            for i in 0..<count {
                let ptr = raw.baseAddress!.advanced(by: offset + i * stride)
                // loadUnaligned: glTF only guarantees 4-byte alignment, and an
                // accessor's byteOffset can land a float3 off a 16-byte boundary.
                out[i] = SIMD3<Float>(ptr.loadUnaligned(fromByteOffset: 0, as: Float.self),
                                      ptr.loadUnaligned(fromByteOffset: 4, as: Float.self),
                                      ptr.loadUnaligned(fromByteOffset: 8, as: Float.self))
            }
        }
        return out
    }

    private static func readVec4(_ doc: Doc, _ bufferData: [Data], _ index: Int) throws -> [SIMD4<Float>] {
        guard doc.accessors[index].componentType == FLOAT,
              doc.accessors[index].type == "VEC4" else {
            throw Error.unsupported("VEC4 accessor that is not float")
        }
        let (bytes, offset, stride, count) = try accessorBytes(doc, bufferData, index, elementSize: 16)
        var out = [SIMD4<Float>](repeating: .zero, count: count)
        bytes.withUnsafeBytes { raw in
            for i in 0..<count {
                let ptr = raw.baseAddress!.advanced(by: offset + i * stride)
                out[i] = SIMD4<Float>(ptr.loadUnaligned(fromByteOffset: 0,  as: Float.self),
                                      ptr.loadUnaligned(fromByteOffset: 4,  as: Float.self),
                                      ptr.loadUnaligned(fromByteOffset: 8,  as: Float.self),
                                      ptr.loadUnaligned(fromByteOffset: 12, as: Float.self))
            }
        }
        return out
    }

    private static func readVec2(_ doc: Doc, _ bufferData: [Data], _ index: Int) throws -> [SIMD2<Float>] {
        guard doc.accessors[index].componentType == FLOAT,
              doc.accessors[index].type == "VEC2" else {
            throw Error.unsupported("VEC2 accessor that is not float")
        }
        let (bytes, offset, stride, count) = try accessorBytes(doc, bufferData, index, elementSize: 8)
        var out = [SIMD2<Float>](repeating: .zero, count: count)
        bytes.withUnsafeBytes { raw in
            for i in 0..<count {
                let ptr = raw.baseAddress!.advanced(by: offset + i * stride)
                out[i] = SIMD2<Float>(ptr.loadUnaligned(fromByteOffset: 0, as: Float.self),
                                      ptr.loadUnaligned(fromByteOffset: 4, as: Float.self))
            }
        }
        return out
    }

    private static func readIndices(_ doc: Doc, _ bufferData: [Data], _ index: Int) throws -> [UInt32] {
        let acc = doc.accessors[index]
        guard acc.type == "SCALAR" else { throw Error.unsupported("non-scalar index accessor") }

        switch acc.componentType {
        case UNSIGNED_SHORT:
            let (bytes, offset, stride, count) = try accessorBytes(doc, bufferData, index, elementSize: 2)
            var out = [UInt32](repeating: 0, count: count)
            bytes.withUnsafeBytes { raw in
                for i in 0..<count {
                    let ptr = raw.baseAddress!.advanced(by: offset + i * stride)
                    out[i] = UInt32(ptr.loadUnaligned(as: UInt16.self))
                }
            }
            return out
        case UNSIGNED_INT:
            let (bytes, offset, stride, count) = try accessorBytes(doc, bufferData, index, elementSize: 4)
            var out = [UInt32](repeating: 0, count: count)
            bytes.withUnsafeBytes { raw in
                for i in 0..<count {
                    let ptr = raw.baseAddress!.advanced(by: offset + i * stride)
                    out[i] = ptr.loadUnaligned(as: UInt32.self)
                }
            }
            return out
        default:
            throw Error.unsupported("index componentType \(acc.componentType)")
        }
    }
}
