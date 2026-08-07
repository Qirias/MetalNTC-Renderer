import Foundation
import Metal
import MetalKit
import simd
import NTCCore
import NTCShared
import AAPLMath

private struct MeshUniforms {
    var mvp:          simd_float4x4
    var model:        simd_float4x4
    var normalMatrix: simd_float4x4
}


private struct TemporalUniforms {
    var viewProj:      simd_float4x4
    var invViewProj:   simd_float4x4
    var deltaRotation: simd_float4x4
    var historyValid:  UInt32
}

/// Mirrors `struct MaterialLayout` in shaders/mesh.metal. -1 means this .ntc
/// does not carry that semantic and the shader falls back to a neutral value.
private struct MaterialLayout {
    var albedo:    Int32 = -1
    var normal:    Int32 = -1
    var roughness: Int32 = -1
    var metalness: Int32 = -1
    var occlusion: Int32 = -1
    var emissive:  Int32 = -1

    /// Semantic names come from the manifest and are written into the .ntc slot
    /// table verbatim, so match on those.
    init(slots: [NTCSlotInfo]) {
        for s in slots {
            let offset = Int32(s.channelOffset)
            switch s.semantic {
            case "Albedo":    albedo    = offset
            case "Normal":    normal    = offset
            case "Roughness": roughness = offset
            case "Metalness": metalness = offset
            case "Occlusion": occlusion = offset
            case "Emissive":  emissive  = offset
            default:          break     // e.g. Displacement, AlphaMask: not shaded yet
            }
        }
    }
}

/// 12 tightly-packed floats, matching Metal's
/// `{ packed_float3 position; packed_float3 normal; packed_float2 uv;
///    packed_float4 tangent; }`. Tangent w is the glTF handedness sign; 0 marks
/// a primitive with no TANGENT so the shader falls back to the geometric normal.
private struct MeshVertex {
    var px, py, pz: Float
    var nx, ny, nz: Float
    var u,  v:      Float
    var tx, ty, tz, tw: Float
}

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    let device: any MTLDevice
    let queue: any MTLCommandQueue

    let meshPSO: any MTLRenderPipelineState
    let depthState: any MTLDepthStencilState


    private var irradianceMap:      (any MTLTexture)? = nil   // diffuse convolution
    private var radianceMap:        (any MTLTexture)? = nil   // prefiltered specular, mipped
    private var brdfLut:            (any MTLTexture)? = nil   // pre-integrated split-sum LUT
    private var environmentCubemap: (any MTLTexture)? = nil   // for the skybox
    private var skyboxPSO:          (any MTLRenderPipelineState)? = nil
    private var skyboxDepthState:   (any MTLDepthStencilState)?   = nil


    private var sceneColorTexture: (any MTLTexture)? = nil   // linear HDR rgb, depth in alpha
    private var sceneDepthTexture: (any MTLTexture)? = nil   // depth test only
    private var historyTextures:   [any MTLTexture]  = []    // 2
    private var historyIndex = 0
    private var historyValid = false
    private var resolvePSO: (any MTLComputePipelineState)? = nil
    private var prevSpin = matrix_identity_float4x4

    /// GPU resources decoded from one .ntc (one material). Shared by every
    /// submesh that references that material.
    fileprivate struct NTCResource {
        let latentTexture: any MTLTexture
        let mlpBuffer:     any MTLBuffer
        let constsBuffer:  any MTLBuffer
        var materialLayout: MaterialLayout
        /// Offset-binary dequant folded into (scale, bias): feature = v*scale + bias.
        var gridDequant: SIMD2<Float>
    }

    /// One drawable primitive: geometry, world transform, and which .ntc
    /// (index into `ntcResources`) decodes its material.
    fileprivate struct DrawSubmesh {
        let vertexBuffer: any MTLBuffer
        let indexBuffer:  any MTLBuffer
        let indexCount:   Int
        let transform:    simd_float4x4
        let ntcIndex:     Int
    }

    private let ntcResources: [NTCResource]
    private let submeshes:    [DrawSubmesh]

    private let benchmark: Bool

    var startTime: CFTimeInterval = CACurrentMediaTime()
    var aspect: Float = 1.0

    /// Translates the model's world-space bounding-box center to the origin, so
    /// it sits at screen center (the camera looks at the origin) and spins about
    /// its own center rather than orbiting.
    private let recenter: simd_float4x4

    private var gpuTimes: [Double] = []

    private var frameIndex: UInt64 = 0

    init(view: MTKView, device: any MTLDevice, gltfURL: URL, hdrURL: URL,
         benchmark: Bool, benchmarkNTC: URL) throws {
        self.device = device
        self.queue  = device.makeCommandQueue()!
        self.benchmark = benchmark

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        let vertexFunction   = library.makeFunction(name: benchmark ? "fullscreen_vs" : "mesh_vs")!
        let fragmentFunction = library.makeFunction(name: benchmark ? "bench_fs" : "mesh_fs")!

        let offscreenColorFormat: MTLPixelFormat = .rgba16Float
        let pipelineDesc = MTLRenderPipelineDescriptor()
        pipelineDesc.vertexFunction   = vertexFunction
        pipelineDesc.fragmentFunction = fragmentFunction
        pipelineDesc.colorAttachments[0].pixelFormat = benchmark ? view.colorPixelFormat : offscreenColorFormat
        pipelineDesc.depthAttachmentPixelFormat      = view.depthStencilPixelFormat
        self.meshPSO = try device.makeRenderPipelineState(descriptor: pipelineDesc)

        // matrix_perspective_right_hand is reverse-Z (near maps to 1, far to 0),
        // so the test is `greater` against a 0.0 clear -- see main.swift.
        let depthDesc = MTLDepthStencilDescriptor()
        depthDesc.depthCompareFunction = .greater
        depthDesc.isDepthWriteEnabled  = true
        self.depthState = device.makeDepthStencilState(descriptor: depthDesc)!

        if benchmark {
            let resource = try Renderer.loadNTCResource(device: device, url: benchmarkNTC)
            self.ntcResources = [resource]
            self.submeshes     = []
            self.recenter      = matrix_identity_float4x4
            super.init()
            self.aspect = Float(view.drawableSize.width / max(view.drawableSize.height, 1))
            print("BENCHMARK: full-screen inference on \(benchmarkNTC.lastPathComponent)")
            return
        }

        // a submesh whose .ntc is missing is skipped so the rest of the model still draws
        let scene = try GLTF.loadScene(at: gltfURL)
        let dir   = gltfURL.deletingLastPathComponent()

        var resources: [NTCResource]   = []
        var indexByName: [String: Int] = [:]
        var draws: [DrawSubmesh]       = []

        var boundsMin = SIMD3<Float>(repeating:  .greatestFiniteMagnitude)
        var boundsMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for submesh in scene.submeshes {
            let ntcIndex: Int
            if let cached = indexByName[submesh.materialName] {
                ntcIndex = cached
            } else {
                let url = dir.appendingPathComponent("\(submesh.materialName).ntc")
                let resource = try Renderer.loadNTCResource(device: device, url: url)
                ntcIndex = resources.count
                resources.append(resource)
                indexByName[submesh.materialName] = ntcIndex
            }

            for position in submesh.positions {
                let world = submesh.transform * SIMD4<Float>(position, 1)
                let p = SIMD3<Float>(world.x, world.y, world.z)
                boundsMin = simd_min(boundsMin, p)
                boundsMax = simd_max(boundsMax, p)
            }

            let vertices = Renderer.buildVertices(submesh)
            let vertexBuffer = device.makeBuffer(bytes: vertices,
                                                 length: vertices.count * MemoryLayout<MeshVertex>.stride,
                                                 options: .storageModeShared)!
            vertexBuffer.label = "\(submesh.materialName).vertices"
            let indexBuffer = device.makeBuffer(bytes: submesh.indices,
                                                length: submesh.indices.count * MemoryLayout<UInt32>.stride,
                                                options: .storageModeShared)!
            indexBuffer.label = "\(submesh.materialName).indices"

            draws.append(DrawSubmesh(vertexBuffer: vertexBuffer, indexBuffer: indexBuffer,
                                     indexCount: submesh.indices.count,
                                     transform: submesh.transform, ntcIndex: ntcIndex))
        }

        guard !draws.isEmpty else {
            throw GLTF.Error.missing("no drawable submesh with a trained .ntc beside \(gltfURL.lastPathComponent)")
        }

        let center = (boundsMin + boundsMax) * 0.5
        var recenterMatrix = matrix_identity_float4x4
        recenterMatrix.columns.3 = SIMD4<Float>(-center.x, -center.y, -center.z, 1)
        self.recenter = recenterMatrix

        self.ntcResources = resources
        self.submeshes     = draws


        let ibl = try Renderer.buildIBL(device: device, queue: queue, library: library, hdrURL: hdrURL)
        self.environmentCubemap = ibl.environment
        self.irradianceMap      = ibl.irradiance
        self.radianceMap        = ibl.radiance
        self.brdfLut            = ibl.brdf

        let skyDesc = MTLRenderPipelineDescriptor()
        skyDesc.vertexFunction   = library.makeFunction(name: "skybox_vs")!
        skyDesc.fragmentFunction = library.makeFunction(name: "skybox_fs")!
        skyDesc.colorAttachments[0].pixelFormat = offscreenColorFormat
        skyDesc.depthAttachmentPixelFormat      = view.depthStencilPixelFormat
        self.skyboxPSO = try device.makeRenderPipelineState(descriptor: skyDesc)

        self.resolvePSO = try device.makeComputePipelineState(
            function: library.makeFunction(name: "temporal_resolve")!)

        // mesh is drawn first (clear depth 0 for reverse-z). The skybox sits at the
        // far plane (z = 0) and tests equal so it survives only on pixels the mesh left
        // at the cleared depth.
        let skyDepthDesc = MTLDepthStencilDescriptor()
        skyDepthDesc.depthCompareFunction = .equal
        skyDepthDesc.isDepthWriteEnabled  = false
        self.skyboxDepthState = device.makeDepthStencilState(descriptor: skyDepthDesc)

        super.init()
        self.aspect = Float(view.drawableSize.width / max(view.drawableSize.height, 1))
        allocateTargets(width: max(Int(view.drawableSize.width), 1),
                        height: max(Int(view.drawableSize.height), 1))
        print("loaded \(draws.count) submesh(es), \(resources.count) material(s)")
    }

    private func allocateTargets(width: Int, height: Int) {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                width: width, height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .private
        let sceneColor = device.makeTexture(descriptor: colorDesc)!
        sceneColor.label = "sceneColor (linear HDR)"
        self.sceneColorTexture = sceneColor

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                                width: width, height: height, mipmapped: false)
        depthDesc.usage = [.renderTarget]
        depthDesc.storageMode = .private
        let sceneDepth = device.makeTexture(descriptor: depthDesc)!
        sceneDepth.label = "sceneDepth"
        self.sceneDepthTexture = sceneDepth

        let histDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                               width: width, height: height, mipmapped: false)
        histDesc.usage = [.shaderRead, .shaderWrite]
        histDesc.storageMode = .private
        self.historyTextures = (0..<2).map { index in
            let texture = device.makeTexture(descriptor: histDesc)!
            texture.label = "history[\(index)]"
            return texture
        }
        self.historyIndex = 0
        self.historyValid = false
    }

    /// Decode one .ntc into GPU resources (latent texture, mlp buffer, consts,
    /// material layout, dequant).
    fileprivate static func loadNTCResource(device: any MTLDevice, url: URL) throws -> NTCResource {
        let ntc     = try readNTC(from: url)
        let buffers = try buildNTCBuffers(device: device, ntc: ntc, lod: 0)
        let bits       = Int(ntc.header.quantBits)
        let quantScale = ntc.header.quantScale
        return NTCResource(latentTexture:  buffers.latents,
                           mlpBuffer:      buffers.mlp,
                           constsBuffer:   buffers.consts,
                           materialLayout: MaterialLayout(slots: ntc.slots),
                           gridDequant:    SIMD2<Float>(Float((1 << bits) - 1) * quantScale,
                                                        -Float(1 << (bits - 1)) * quantScale))
    }

    // MARK: Image-based lighting

    fileprivate struct IBLTextures {
        let environment: any MTLTexture
        let irradiance:  any MTLTexture
        let radiance:    any MTLTexture
        let brdf:        any MTLTexture
    }

    fileprivate static func buildIBL(device:  any MTLDevice,
                                     queue:   any MTLCommandQueue,
                                     library: any MTLLibrary,
                                     hdrURL:  URL) throws -> IBLTextures {
        let hdrImage = try HDRLoader.load(url: hdrURL)
        let equirectDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float,
                                                                    width:  hdrImage.width,
                                                                    height: hdrImage.height,
                                                                    mipmapped: false)
        equirectDesc.usage = .shaderRead
        equirectDesc.storageMode = .shared
        let equirect = device.makeTexture(descriptor: equirectDesc)!
        equirect.label = "HDR equirect"
        hdrImage.rgba.withUnsafeBytes { rawPixels in
            equirect.replace(region: MTLRegionMake2D(0, 0, hdrImage.width, hdrImage.height), mipmapLevel: 0,
                             withBytes: rawPixels.baseAddress!,
                             bytesPerRow: hdrImage.width * 4 * MemoryLayout<Float>.stride)
        }

        func makeComputePSO(_ functionName: String) throws -> any MTLComputePipelineState {
            try device.makeComputePipelineState(function: library.makeFunction(name: functionName)!)
        }
        let equirectPSO   = try makeComputePSO("equirect_to_cubemap")
        let irradiancePSO = try makeComputePSO("convolve_irradiance")
        let radiancePSO   = try makeComputePSO("convolve_radiance")

        func makeCubemap(size: Int, mipCount: Int, label: String) -> any MTLTexture {
            let descriptor = MTLTextureDescriptor.textureCubeDescriptor(pixelFormat: .rgba16Float,
                                                                        size: size, mipmapped: mipCount > 1)
            descriptor.mipmapLevelCount = mipCount
            descriptor.usage = [.shaderWrite, .shaderRead]
            descriptor.storageMode = .private
            let texture = device.makeTexture(descriptor: descriptor)!
            texture.label = label
            return texture
        }
        let environmentSize = 1024, irradianceSize = 32, radianceSize = 512
        let radianceMipCount = Int(log2(Double(radianceSize))) + 1   // 10: 512, 256, ..., 1
        let environment = makeCubemap(size: environmentSize, mipCount: 1,                label: "IBL.environment")
        let irradiance  = makeCubemap(size: irradianceSize,  mipCount: 1,                label: "IBL.irradiance")
        let radiance    = makeCubemap(size: radianceSize,    mipCount: radianceMipCount, label: "IBL.radiance")

        func dispatchCube(_ encoder: any MTLComputeCommandEncoder, faceSize: Int) {
            let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
            let groupCount = MTLSize(width: (faceSize + 7) / 8, height: (faceSize + 7) / 8, depth: 6)
            encoder.dispatchThreadgroups(groupCount, threadsPerThreadgroup: threadsPerGroup)
        }

        let commandBuffer = queue.makeCommandBuffer()!
        commandBuffer.label = "IBL precompute"

        // equirect -> environment cubemap
        do {
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.label = "equirect_to_cubemap"
            encoder.setComputePipelineState(equirectPSO)
            encoder.setTexture(equirect, index: 0)
            encoder.setTexture(environment, index: 1)
            dispatchCube(encoder, faceSize: environmentSize)
            encoder.endEncoding()
        }
        // environment -> diffuse irradiance
        do {
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.label = "convolve_irradiance"
            encoder.setComputePipelineState(irradiancePSO)
            encoder.setTexture(environment, index: 0)
            encoder.setTexture(irradiance, index: 1)
            dispatchCube(encoder, faceSize: irradianceSize)
            encoder.endEncoding()
        }
        // environment -> prefiltered specular, one dispatch per roughness mip
        do {
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.label = "convolve_radiance"
            encoder.setComputePipelineState(radiancePSO)
            encoder.setTexture(environment, index: 0)
            for mipLevel in 0..<radianceMipCount {
                var roughness = Float(mipLevel) / Float(radianceMipCount - 1)
                let mipView = radiance.makeTextureView(pixelFormat: .rgba16Float,
                                                       textureType: .typeCube,
                                                       levels: mipLevel..<(mipLevel + 1),
                                                       slices: 0..<6)!
                mipView.label = "IBL.radiance.mip\(mipLevel)"
                encoder.setTexture(mipView, index: 1)
                encoder.setBytes(&roughness, length: MemoryLayout<Float>.size, index: 0)
                dispatchCube(encoder, faceSize: radianceSize >> mipLevel)
            }
            encoder.endEncoding()
        }

        let brdfSize = 512
        let brdfDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float,
                                                                width: brdfSize, height: brdfSize,
                                                                mipmapped: false)
        brdfDesc.usage = [.shaderWrite, .shaderRead]
        brdfDesc.storageMode = .private
        let brdf = device.makeTexture(descriptor: brdfDesc)!
        brdf.label = "IBL.brdfLut"
        do {
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.label = "integrate_brdf"
            encoder.setComputePipelineState(try makeComputePSO("integrate_brdf"))
            encoder.setTexture(brdf, index: 0)
            let threadsPerGroup = MTLSize(width: 8, height: 8, depth: 1)
            let groupCount = MTLSize(width: (brdfSize + 7) / 8, height: (brdfSize + 7) / 8, depth: 1)
            encoder.dispatchThreadgroups(groupCount, threadsPerThreadgroup: threadsPerGroup)
            encoder.endEncoding()
        }

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()

        print("IBL: env, irradiance, radiance (10 mips), BRDF LUT computed")
        return IBLTextures(environment: environment, irradiance: irradiance, radiance: radiance, brdf: brdf)
    }

    private static func buildVertices(_ submesh: GLTF.Submesh) -> [MeshVertex] {
        var vertices = [MeshVertex]()
        vertices.reserveCapacity(submesh.positions.count)
        for i in 0..<submesh.positions.count {
            let position = submesh.positions[i]
            let normal   = i < submesh.normals.count ? submesh.normals[i] : SIMD3<Float>(0, 1, 0)
            let uv       = i < submesh.uvs.count     ? submesh.uvs[i]     : SIMD2<Float>(0, 0)
            // Zero tangent (w = 0) signals "no TANGENT" to the shader.
            let tangent  = i < submesh.tangents.count ? submesh.tangents[i] : SIMD4<Float>(0, 0, 0, 0)
            vertices.append(MeshVertex(px: position.x, py: position.y, pz: position.z,
                                       nx: normal.x,   ny: normal.y,   nz: normal.z,
                                       u:  uv.x,       v:  uv.y,
                                       tx: tangent.x,  ty: tangent.y, tz: tangent.z, tw: tangent.w))
        }
        return vertices
    }

    // MARK: MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        self.aspect = Float(size.width / max(size.height, 1))
        if !benchmark {
            allocateTargets(width: max(Int(size.width), 1), height: max(Int(size.height), 1))
        }
    }

    func draw(in view: MTKView) {
        guard let drawable = view.currentDrawable else { return }

        let commandBuffer = queue.makeCommandBuffer()!
        var frame = UInt32(truncatingIfNeeded: frameIndex)

        if benchmark {
            guard let renderPassDesc = view.currentRenderPassDescriptor else { return }
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc)!
            encoder.setRenderPipelineState(meshPSO)
            encoder.setDepthStencilState(depthState)
            var resource = ntcResources[0]
            encoder.setCullMode(.none)
            encoder.setFragmentTexture(resource.latentTexture, index: 0)
            encoder.setFragmentBuffer(resource.mlpBuffer,    offset: 0, index: 1)
            encoder.setFragmentBuffer(resource.constsBuffer, offset: 0, index: 2)
            encoder.setFragmentBytes(&resource.materialLayout,
                                     length: MemoryLayout<MaterialLayout>.stride, index: 3)
            encoder.setFragmentBytes(&resource.gridDequant,
                                     length: MemoryLayout<SIMD2<Float>>.stride, index: 4)
            encoder.setFragmentBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        } else {
            guard let sceneColor = sceneColorTexture,
                  let sceneDepth = sceneDepthTexture,
                  let resolvePSO = resolvePSO,
                  historyTextures.count == 2 else { return }

            let camEye = SIMD3<Float>(0, 0.0, 0.8)
            let time = Float(CACurrentMediaTime() - startTime)
            let spin = matrix4x4_rotation(time * 0.7, 0, 1, 0)
            let viewMatrix = matrix_look_at_right_hand(camEye,
                                                       SIMD3<Float>(0, 0, 0),
                                                       SIMD3<Float>(0, 1, 0))
            let projMatrix = matrix_perspective_right_hand(radians_from_degrees(60),
                                                           aspect, 0.1, 100.0)
            let viewProj = projMatrix * viewMatrix

            // pass 1: mesh + skybox into the offscreen linear-HDR buffer
            let passDesc = MTLRenderPassDescriptor()
            passDesc.colorAttachments[0].texture     = sceneColor
            passDesc.colorAttachments[0].loadAction  = .clear
            passDesc.colorAttachments[0].clearColor  = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
            passDesc.colorAttachments[0].storeAction = .store
            passDesc.depthAttachment.texture     = sceneDepth
            passDesc.depthAttachment.loadAction  = .clear
            passDesc.depthAttachment.clearDepth  = 0.0
            passDesc.depthAttachment.storeAction = .dontCare
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: passDesc)!

            encoder.setRenderPipelineState(meshPSO)
            encoder.setDepthStencilState(depthState)
            encoder.setCullMode(.back)
            encoder.setFrontFacing(.counterClockwise)

            var camPos = camEye
            for submesh in submeshes {
                let model = spin * recenter * submesh.transform
                var uniforms = MeshUniforms(mvp: viewProj * model,
                                            model: model,
                                            normalMatrix: matrix_inverse_transpose(model))
                var resource = ntcResources[submesh.ntcIndex]   // value copy; setFragmentBytes needs inout

                encoder.setVertexBuffer(submesh.vertexBuffer, offset: 0, index: 0)
                encoder.setVertexBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 1)
                encoder.setFragmentTexture(resource.latentTexture, index: 0)
                encoder.setFragmentTexture(irradianceMap!, index: 1)
                encoder.setFragmentTexture(radianceMap!,   index: 2)
                encoder.setFragmentTexture(brdfLut!,       index: 3)
                encoder.setFragmentBuffer(resource.mlpBuffer,    offset: 0, index: 1)
                encoder.setFragmentBuffer(resource.constsBuffer, offset: 0, index: 2)
                encoder.setFragmentBytes(&resource.materialLayout,
                                         length: MemoryLayout<MaterialLayout>.stride, index: 3)
                encoder.setFragmentBytes(&resource.gridDequant,
                                         length: MemoryLayout<SIMD2<Float>>.stride, index: 4)
                encoder.setFragmentBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 5)
                encoder.setFragmentBytes(&camPos, length: MemoryLayout<SIMD3<Float>>.stride, index: 6)
                encoder.drawIndexedPrimitives(type: .triangle,
                                              indexCount: submesh.indexCount,
                                              indexType: .uint32,
                                              indexBuffer: submesh.indexBuffer,
                                              indexBufferOffset: 0)
            }

            encoder.setRenderPipelineState(skyboxPSO!)
            encoder.setDepthStencilState(skyboxDepthState!)
            encoder.setCullMode(.none)
            var invViewProj = simd_inverse(viewProj)
            encoder.setFragmentTexture(environmentCubemap!, index: 0)
            encoder.setFragmentBytes(&invViewProj, length: MemoryLayout<simd_float4x4>.stride, index: 0)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()

            // pass 2: temporal resolve
            let readIdx  = historyIndex
            let writeIdx = 1 - historyIndex
            var uniforms = TemporalUniforms(viewProj:      viewProj,
                                            invViewProj:   invViewProj,
                                            deltaRotation: prevSpin * simd_inverse(spin),
                                            historyValid:  historyValid ? 1 : 0)

            let resolve = commandBuffer.makeComputeCommandEncoder()!
            resolve.setComputePipelineState(resolvePSO)
            resolve.setTexture(sceneColor,               index: 0)
            resolve.setTexture(historyTextures[readIdx], index: 1)
            resolve.setTexture(drawable.texture,         index: 2)
            resolve.setTexture(historyTextures[writeIdx], index: 3)
            resolve.setBytes(&uniforms, length: MemoryLayout<TemporalUniforms>.stride, index: 0)
            let groupSize = MTLSize(width: 8, height: 8, depth: 1)
            let groupCount = MTLSize(width:  (sceneColor.width  + 7) / 8,
                                     height: (sceneColor.height + 7) / 8, depth: 1)
            resolve.dispatchThreadgroups(groupCount, threadsPerThreadgroup: groupSize)
            resolve.endEncoding()

            prevSpin     = spin
            historyIndex = writeIdx
            historyValid = true
        }

        commandBuffer.addCompletedHandler { [weak self] completed in
            let frameMs = (completed.gpuEndTime - completed.gpuStartTime) * 1000.0
            Task { @MainActor in self?.recordFrame(frameMs) }
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()

        frameIndex += 1
    }

    private func recordFrame(_ frameMs: Double) {
        gpuTimes.append(frameMs)
        guard gpuTimes.count % 60 == 0 else { return }
        let recent = gpuTimes.suffix(60)
        print(String(format: "frame %4d  gpu %.2f ms (last 60 avg)", gpuTimes.count, recent.reduce(0, +) / Double(recent.count)))
        fflush(stdout)
    }

    // MARK: .ntc -> GPU buffers

    /// Upload the .ntc payload to GPU:
    /// - `latents`: abgr4Unorm texture2d_array. Grid index -> mip level, every 4
    ///              features -> one array slice. The offset-binary 4-bit ints go
    ///              in verbatim; the fragment shader's fixed-function sampler does
    ///              bilinear filtering and dequantizes to `(15*v - 8) * q`.
    /// - `mlp`    : IEEE 754 binary16 weights/biases
    /// - `consts` : StepConstants block driving `ntc_decode_quant`
    /// MLP offsets are re-baselined to start at 0 (relative to the MLP
    /// buffer) since it lives in its own MTLBuffer here.
    static func buildNTCBuffers(device: any MTLDevice,
                                ntc:    NTCFile,
                                lod:    Int) throws -> (latents: any MTLTexture,
                                                        mlp:     any MTLBuffer,
                                                        consts:  any MTLBuffer) {
        let kGrids   = Int(ntc.header.kGrids)
        let fPerGrid = Int(ntc.header.fPerGrid)
        let kHidden  = Int(ntc.header.kHidden)
        let kOutMax  = Int(ntc.header.kOutMax)
        let kOut     = Int(ntc.header.kOut)
        let peWaves  = Int(ntc.header.peWaves)
        let fInRaw   = 2 * fPerGrid + 4 * peWaves + 1
        let fIn      = ((fInRaw + 15) / 16) * 16

        var pyramidOffsets = [Int](repeating: 0, count: kGrids + 1)
        for i in 0..<kGrids {
            let size = Int(ntc.pyramidSizes[i])
            pyramidOffsets[i + 1] = pyramidOffsets[i] + size * size * fPerGrid
        }
        let gridCount = pyramidOffsets[kGrids]
        precondition(ntc.grid.count == gridCount, "grid byte count \(ntc.grid.count) != expected \(gridCount)")

        let mlpOffW1 = 0
        let mlpOffB1 = mlpOffW1 + fIn * kHidden
        let mlpOffW2 = mlpOffB1 + kHidden
        let mlpOffB2 = mlpOffW2 + kHidden * kHidden
        let mlpOffW3 = mlpOffB2 + kHidden
        let mlpOffB3 = mlpOffW3 + kHidden * kOutMax
        let mlpCount = mlpOffB3 + kOutMax
        precondition(ntc.mlp.count == mlpCount,
                     "mlp half count \(ntc.mlp.count) != expected \(mlpCount)")

        let latentTex = buildLatentTexture(device: device,
                                           grid: ntc.grid,
                                           pyramidSizes: ntc.pyramidSizes.map { Int($0) },
                                           pyramidOffsets: pyramidOffsets,
                                           fPerGrid: fPerGrid,
                                           bits: Int(ntc.header.quantBits))

        let mlpBuf = ntc.mlp.withUnsafeBufferPointer { buf in
            device.makeBuffer(bytes: buf.baseAddress!,
                              length: mlpCount * MemoryLayout<UInt16>.stride,
                              options: .storageModeShared)!
        }
        mlpBuf.label = "NTC.mlp (fp16, packed)"

        let constsBuf = buildStepConstantsBuffer(device: device,
                                                 ntc: ntc,
                                                 pyramidOffsets: pyramidOffsets,
                                                 mlpOffW1: mlpOffW1, mlpOffB1: mlpOffB1,
                                                 mlpOffW2: mlpOffW2, mlpOffB2: mlpOffB2,
                                                 mlpOffW3: mlpOffW3, mlpOffB3: mlpOffB3,
                                                 mlpCount: mlpCount,
                                                 kOut: kOut,
                                                 lod: lod)
        return (latentTex, mlpBuf, constsBuf)
    }

    /// Pack the offset-binary grid ints into a texture2d_array so the fragment
    /// shader can bilinear-filter them for free (the shader's scale/bias dequant
    /// is bit-depth agnostic; see `sample_latent_grid`).
    ///
    /// `bits` selects the storage format: 4 -> `abgr4Unorm` (4 packed/texel),
    /// 8 -> `rgba8Unorm` (4 bytes/texel). Both are filterable on Apple GPU.
    ///
    /// The grids form a power-of-two pyramid (each half the previous), so grid
    /// index `m` becomes texture mip `m` (size `pyramidSizes[m]`). Each grid has
    /// `fPerGrid` features; 4 features pack into the RGBA of one array slice, so
    /// `arrayLength = fPerGrid / 4`. Feature `4*slice+c` goes into channel `c`,
    /// matching the `.rgba` read order in `sample_latent_grid`. 
    private static func buildLatentTexture(device:         any MTLDevice,
                                           grid:           [UInt8],
                                           pyramidSizes:   [Int],
                                           pyramidOffsets: [Int],
                                           fPerGrid:       Int,
                                           bits:           Int) -> any MTLTexture {
        let kGrids = pyramidSizes.count
        precondition(fPerGrid % 4 == 0, "fPerGrid \(fPerGrid) must be a multiple of 4")
        let slices   = fPerGrid / 4
        let baseSize = pyramidSizes[0]

        let texDesc = MTLTextureDescriptor()
        texDesc.textureType      = .type2DArray
        switch bits {
            case 4: texDesc.pixelFormat = .abgr4Unorm
            case 8: texDesc.pixelFormat = .rgba8Unorm
            default: fatalError("unsupported quantBits \(bits); expected 4 or 8")
        }
        texDesc.width            = baseSize
        texDesc.height           = baseSize
        texDesc.arrayLength      = slices
        texDesc.mipmapLevelCount = kGrids
        texDesc.usage            = .shaderRead
        texDesc.storageMode      = .shared
        let texture = device.makeTexture(descriptor: texDesc)!
        texture.label = "NTC.latents (\(bits)-bit)"

        for mip in 0..<kGrids {
            let width = pyramidSizes[mip]
            precondition(width == max(baseSize >> mip, 1), "grid \(mip) size \(width) is not mip \(mip) of \(baseSize); pyramid must be power-of-two")
            let gridBase = pyramidOffsets[mip]
            for slice in 0..<slices {
                let region = MTLRegionMake2D(0, 0, width, width)
                if bits == 4 {
                    var texels = [UInt16](repeating: 0, count: width * width)
                    for y in 0..<width {
                        for x in 0..<width {
                            let srcIndex = (y * width + x) * fPerGrid + gridBase + slice * 4
                            let nib0 = UInt16(grid[srcIndex + 0] & 0x0F)   // -> .r (high)
                            let nib1 = UInt16(grid[srcIndex + 1] & 0x0F)   // -> .g
                            let nib2 = UInt16(grid[srcIndex + 2] & 0x0F)   // -> .b
                            let nib3 = UInt16(grid[srcIndex + 3] & 0x0F)   // -> .a (low)
                            texels[y * width + x] = (nib0 << 12) | (nib1 << 8) | (nib2 << 4) | nib3
                        }
                    }
                    texels.withUnsafeBytes { raw in
                        texture.replace(region: region, mipmapLevel: mip, slice: slice,
                                        withBytes: raw.baseAddress!,
                                        bytesPerRow: width * 2, bytesPerImage: width * width * 2)
                    }
                } else {
                    var texels = [UInt8](repeating: 0, count: width * width * 4)
                    for y in 0..<width {
                        for x in 0..<width {
                            let srcIndex = (y * width + x) * fPerGrid + gridBase + slice * 4
                            let dstIndex = (y * width + x) * 4
                            texels[dstIndex + 0] = grid[srcIndex + 0]   // .r
                            texels[dstIndex + 1] = grid[srcIndex + 1]   // .g
                            texels[dstIndex + 2] = grid[srcIndex + 2]   // .b
                            texels[dstIndex + 3] = grid[srcIndex + 3]   // .a
                        }
                    }
                    texels.withUnsafeBytes { raw in
                        texture.replace(region: region, mipmapLevel: mip, slice: slice,
                                        withBytes: raw.baseAddress!,
                                        bytesPerRow: width * 4, bytesPerImage: width * width * 4)
                    }
                }
            }
        }
        return texture
    }

    /// Build the `StepConstants` MTLBuffer for the fragment shader.
    ///
    /// The C `struct StepConstants` in NTCCore/shaders/common.h is a sequence
    /// of 4-byte fields (uint or float), some scalar, some fixed-size arrays.
    /// We fill the buffer by walking a byte cursor field-by-field in
    /// declaration order. Field order is load-bearing -- if you edit that
    /// struct, edit this function to match.
    private static func buildStepConstantsBuffer(device:         any MTLDevice,
                                                 ntc:            NTCFile,
                                                 pyramidOffsets: [Int],
                                                 mlpOffW1: Int, mlpOffB1: Int,
                                                 mlpOffW2: Int, mlpOffB2: Int,
                                                 mlpOffW3: Int, mlpOffB3: Int,
                                                 mlpCount: Int,
                                                 kOut:     Int,
                                                 lod:      Int) -> any MTLBuffer {
        let kGridsC   = K_GRIDS       // 8
        let maxLodsC  = MAX_LODS      // 13
        let kOutMaxC  = K_OUT_MAX     // 16

        let slotCount =
            1                                       // kBatch
          + (kGridsC + 1) + kGridsC                 // pyramid offsets + sizes
          + 6                                       // offsetW1..offsetB3
          + 1 + 1                                   // total, bits
          + 3                                       // q, lo, hi
          + 1 + 1                                   // adamOffset, inferLod
          + maxLodsC                                // neuralMipForLod
          + 1 + 1                                   // kOut, nSlices
          + kOutMaxC + kOutMaxC                     // sliceChannels/Offsets
          + 3 + 1                                   // srcW, srcH, mipCount, posScale

        let sizeBytes = slotCount * 4
        let buf = device.makeBuffer(length: sizeBytes, options: .storageModeShared)!
        buf.label = "NTC.stepConsts"
        let base = buf.contents()
        memset(base, 0, sizeBytes)

        var cursor = 0
        func u32(_ v: UInt32) {
            base.storeBytes(of: v, toByteOffset: cursor, as: UInt32.self)
            cursor += 4
        }
        func f32(_ v: Float) {
            base.storeBytes(of: v, toByteOffset: cursor, as: Float.self)
            cursor += 4
        }
        func u32Array(_ src: [UInt32], length: Int) {
            for i in 0..<length { u32(i < src.count ? src[i] : 0) }
        }

        let srcW     = ntc.header.srcW
        let mipCount = Int(ntc.header.mipCount)

        u32(0)                                                                // kBatch
        u32Array(pyramidOffsets.map { UInt32($0) },      length: kGridsC + 1) // pyramidOffsets
        u32Array(ntc.pyramidSizes,                       length: kGridsC)     // pyramidSizes
        u32(UInt32(mlpOffW1)); u32(UInt32(mlpOffB1))                          // W1, B1
        u32(UInt32(mlpOffW2)); u32(UInt32(mlpOffB2))                          // W2, B2
        u32(UInt32(mlpOffW3)); u32(UInt32(mlpOffB3))                          // W3, B3
        u32(UInt32(mlpCount))                                                 // total
        u32(ntc.header.quantBits)                                             // bits
        f32(ntc.header.quantScale)                                            // q
        f32(0); f32(0)                                                        // lo, hi
        u32(0)                                                                // adamOffset
        u32(UInt32(lod))                                                      // inferLod
        // MAX_LODS is 13, pad the missing LODs with the finest LOD.
        let nmTail = ntc.neuralMipsForLod.last ?? 0
        u32Array(ntc.neuralMipsForLod + Array(repeating: nmTail, count: max(0, maxLodsC - mipCount)),
                 length: maxLodsC)                                            // neuralMipForLod
        u32(UInt32(kOut))                                                     // kOut
        u32(UInt32(ntc.slots.count))                                          // nSlices
        u32Array(ntc.slots.map { UInt32($0.channels)      }, length: kOutMaxC)
        u32Array(ntc.slots.map { UInt32($0.channelOffset) }, length: kOutMaxC)
        u32(srcW)                                                             // srcW
        u32(ntc.header.srcH)                                                  // srcH
        u32(UInt32(mipCount))                                                 // mipCount
        f32(Float(srcW) / 8.0)                                                // posScale

        precondition(cursor == sizeBytes, "StepConstants layout drift: wrote \(cursor) bytes, expected \(sizeBytes)")
        return buf
    }
}
