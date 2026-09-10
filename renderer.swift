// MetalNTC Renderer — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

import Foundation
import AppKit
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
    private var blueNoise:          (any MTLTexture)? = nil   // 64x64 STF dither
    private var skyboxDepthState:   (any MTLDepthStencilState)?   = nil


    private var sceneColorTexture: (any MTLTexture)? = nil   // linear HDR rgb, depth in alpha
    private var sceneDepthTexture: (any MTLTexture)? = nil   // depth test, then read by the decode pass

    private var gbufferUvLod:   (any MTLTexture)? = nil   // uv.xy, lod, material index
    private var gbufferNormal:  (any MTLTexture)? = nil   // world normal.xyz, tangent sign
    private var gbufferTangent: (any MTLTexture)? = nil   // world tangent.xyz
    private var gbufferPSO: (any MTLRenderPipelineState)? = nil
    private var decodePSO:  (any MTLComputePipelineState)? = nil
    /// Benchmark-mode twin of decodePSO: full-screen tensor ops decode, directly
    /// comparable to the benchmark_fs render pass.
    private var benchmarkDecodePSO: (any MTLComputePipelineState)? = nil
    private var historyTextures:   [any MTLTexture]  = []    // 2
    private var historyIndex = 0
    private var historyValid = false
    private var resolvePSO: (any MTLComputePipelineState)? = nil
    private var prevSpin = matrix_identity_float4x4

    /// One entry per material in the slot buffer the decode reads.
    /// Mirrors  NTCMaterialSlot in mesh.metal
    fileprivate struct NTCMaterialSlot {
        var latents:     MTLResourceID
        var mlp:         UInt64
        var consts:      UInt64
        var layout:      MaterialLayout
        var gridDequant: SIMD2<Float>
    }

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

    private let ntcVariants: [(quality: Quality, resources: [NTCResource], slots: any MTLBuffer)]
    private var activeVariant: Int

    var availableQualities: [Quality] { ntcVariants.map(\.quality) }
    var activeQualityIndex: Int { activeVariant }

    private let submeshes:    [DrawSubmesh]

    private let benchmark: Bool
    private let benchmarkTensorOps: Bool

    var startTime: CFTimeInterval = CACurrentMediaTime()
    var aspect: Float = 1.0

    private let recenter: simd_float4x4

    private var gpuTimes: [Double] = []
    private var lastDrawableSize = CGSize(width: 1, height: 1)

    private var frameIndex: UInt64 = 0

    // MARK: Render toggles / camera state

    private enum RenderFlags {
        static let stochasticLOD: UInt32 = 1 << 0
    }

    var stochasticLOD = true
    var temporalAccumulation = true
    var autoRotate = true

    private var yaw: Float = 0
    private var pitch: Float = 0
    private var cameraDistance: Float = 0.8
    private let cameraDistanceRange: ClosedRange<Float> = 0.15...5.0

    func rotate(deltaX: Float, deltaY: Float) {
        guard !autoRotate else { return }
        yaw += deltaX * 0.01
        // clamp just shy of the poles so the up vector never degenerates
        pitch = min(max(pitch + deltaY * 0.01, -1.5), 1.5)
    }

    func zoom(delta: Float) {
        let updated = cameraDistance * exp(-delta * 0.05)   // exponential: even feel at any distance
        cameraDistance = min(max(updated, cameraDistanceRange.lowerBound), cameraDistanceRange.upperBound)
        // the camera MOVED, so the accumulated history was rendered from a
        // different viewpoint; delta-rotation reprojection cannot map it
        historyValid = false
    }

    @objc func stochasticLODChanged(_ sender: NSButton) { stochasticLOD = sender.state == .on }
    @objc func temporalChanged(_ sender: NSButton) {
        temporalAccumulation = sender.state == .on
        historyValid = false
    }
    @objc func autoRotateChanged(_ sender: NSButton) {
        autoRotate = sender.state == .on
        // resync so handing control back to the clock does not jump
        if autoRotate { startTime = CACurrentMediaTime() - Double(yaw / 0.7) }
    }

    init(view: MTKView, device: any MTLDevice, gltfURL: URL, hdrURL: URL,
         blueNoiseURL: URL, benchmark: Bool, benchmarkTensorOps: Bool,
         benchmarkNTC: URL) throws {
        self.device = device
        self.queue  = device.makeCommandQueue()!
        self.benchmark = benchmark
        self.benchmarkTensorOps = benchmarkTensorOps

        let library = try device.makeDefaultLibrary(bundle: Bundle.module)
        let vertexFunction   = library.makeFunction(name: benchmark ? "fullscreen_vs" : "mesh_vs")!
        let fragmentFunction = library.makeFunction(name: benchmark ? "benchmark_fs" : "mesh_fs")!

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

        self.blueNoise = try Renderer.loadBlueNoise(device: device, url: blueNoiseURL)

        if benchmark {
            let resource = try Renderer.loadNTCResource(device: device, url: benchmarkNTC)
            self.ntcVariants   = [(.high, [resource],
                                   Renderer.makeSlotBuffer(device: device, resources: [resource]))]
            self.activeVariant = 0
            self.submeshes     = []
            self.recenter      = matrix_identity_float4x4
            if let benchmarkDecode = library.makeFunction(name: "benchmark_decode") {
                self.benchmarkDecodePSO = try device.makeComputePipelineState(function: benchmarkDecode)
            }
            super.init()
            self.aspect = Float(view.drawableSize.width / max(view.drawableSize.height, 1))
            let usingTensorOps = benchmarkTensorOps && benchmarkDecodePSO != nil
            let path = usingTensorOps ? "tensor ops (benchmark_decode)"
                                      : "half4 per fragment (benchmark_fs)"
            print("BENCHMARK: full-screen inference on \(benchmarkNTC.lastPathComponent) -- \(path)")
            return
        }

        // a submesh whose .ntc is missing is skipped so the rest of the model still draws
        let scene = try GLTF.loadScene(at: gltfURL)
        let dir   = gltfURL.deletingLastPathComponent()

        // materials in first-seen order, keeping only those trained at some
        // quality; a submesh whose material has none is skipped below
        var materialNames: [String]    = []
        var indexByName: [String: Int] = [:]
        for submesh in scene.submeshes where indexByName[submesh.materialName] == nil {
            let trained = Quality.allCases.contains {
                Renderer.ntcURL(dir: dir, material: submesh.materialName, quality: $0) != nil
            }
            guard trained else { continue }
            indexByName[submesh.materialName] = materialNames.count
            materialNames.append(submesh.materialName)
        }

        var variants: [(quality: Quality, resources: [NTCResource], slots: any MTLBuffer)] = []
        for quality in Quality.allCases {
            let urls = materialNames.map { Renderer.ntcURL(dir: dir, material: $0, quality: quality) }
            guard urls.allSatisfy({ $0 != nil }) else { continue }
            let resources = try urls.map { try Renderer.loadNTCResource(device: device, url: $0!) }
            variants.append((quality, resources,
                             Renderer.makeSlotBuffer(device: device, resources: resources)))
        }
        guard !variants.isEmpty else {
            throw GLTF.Error.missing("no trained .ntc beside \(gltfURL.lastPathComponent)")
        }
        self.ntcVariants = variants
        // open on .high when it is there, else the best that is
        self.activeVariant = variants.firstIndex { $0.quality == .high } ?? variants.count - 1

        var draws: [DrawSubmesh] = []

        var boundsMin = SIMD3<Float>(repeating:  .greatestFiniteMagnitude)
        var boundsMax = SIMD3<Float>(repeating: -.greatestFiniteMagnitude)

        for submesh in scene.submeshes {
            guard let ntcIndex = indexByName[submesh.materialName] else { continue }

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

        if let gbufferFunction = library.makeFunction(name: "gbuffer_fs"),
           let decodeFunction  = library.makeFunction(name: "ntc_decode_shade") {
            let gbufferDesc = MTLRenderPipelineDescriptor()
            gbufferDesc.vertexFunction   = library.makeFunction(name: "mesh_vs")!
            gbufferDesc.fragmentFunction = gbufferFunction
            gbufferDesc.colorAttachments[0].pixelFormat = Renderer.gbufferUvLodFormat
            gbufferDesc.colorAttachments[1].pixelFormat = Renderer.gbufferNormalFormat
            gbufferDesc.colorAttachments[2].pixelFormat = Renderer.gbufferTangentFormat
            gbufferDesc.depthAttachmentPixelFormat      = view.depthStencilPixelFormat
            self.gbufferPSO = try device.makeRenderPipelineState(descriptor: gbufferDesc)
            self.decodePSO  = try device.makeComputePipelineState(function: decodeFunction)
        }

        // mesh is drawn first (clear depth 0 for reverse-z); the skybox sits at the
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
        let qualityList = variants.map(\.quality.rawValue).joined(separator: ", ")
        print("loaded \(draws.count) submesh(es), \(materialNames.count) material(s); qualities: \(qualityList)")
    }

    fileprivate static let gbufferUvLodFormat:   MTLPixelFormat = .rgba32Float
    fileprivate static let gbufferNormalFormat:  MTLPixelFormat = .rgba16Float
    fileprivate static let gbufferTangentFormat: MTLPixelFormat = .rgba16Float


    fileprivate static let decodeTileSide = 8
    fileprivate static let decodeTileThreads = 128

    private func allocateTargets(width: Int, height: Int) {
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float,
                                                                width: width, height: height, mipmapped: false)
        colorDesc.usage = [.renderTarget, .shaderRead, .shaderWrite]
        colorDesc.storageMode = .private
        let sceneColor = device.makeTexture(descriptor: colorDesc)!
        sceneColor.label = "sceneColor (linear HDR)"
        self.sceneColorTexture = sceneColor

        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float,
                                                                width: width, height: height, mipmapped: false)
        depthDesc.usage = [.renderTarget, .shaderRead]
        depthDesc.storageMode = .private
        let sceneDepth = device.makeTexture(descriptor: depthDesc)!
        sceneDepth.label = "sceneDepth"
        self.sceneDepthTexture = sceneDepth

        func makeGBuffer(_ format: MTLPixelFormat, _ label: String) -> any MTLTexture {
            let desc = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format,
                                                               width: width, height: height,
                                                               mipmapped: false)
            desc.usage = [.renderTarget, .shaderRead]
            desc.storageMode = .private
            let texture = device.makeTexture(descriptor: desc)!
            texture.label = label
            return texture
        }
        self.gbufferUvLod   = makeGBuffer(Renderer.gbufferUvLodFormat,   "gbuffer uv/lod/material")
        self.gbufferNormal  = makeGBuffer(Renderer.gbufferNormalFormat,  "gbuffer normal")
        self.gbufferTangent = makeGBuffer(Renderer.gbufferTangentFormat, "gbuffer tangent")

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

    /// Packs every material into one buffer the decode can index, so a single
    /// dispatch reaches all of them and each threadgroup loops over only the
    /// materials in its own tile.
    fileprivate static func makeSlotBuffer(device: any MTLDevice,
                                           resources: [NTCResource]) -> any MTLBuffer {
        // the shader static_asserts the same number
        precondition(MemoryLayout<NTCMaterialSlot>.stride == 56, "NTCMaterialSlot must stay in step with mesh.metal")
        var slots = resources.map { resource in
            NTCMaterialSlot(latents: resource.latentTexture.gpuResourceID,
                            mlp:     resource.mlpBuffer.gpuAddress,
                            consts:  resource.constsBuffer.gpuAddress,
                            layout:  resource.materialLayout,
                            gridDequant: resource.gridDequant)
        }
        let buffer = device.makeBuffer(bytes: &slots,
                                       length: MemoryLayout<NTCMaterialSlot>.stride * slots.count,
                                       options: .storageModeShared)!
        buffer.label = "NTC material slots"
        return buffer
    }

    /// Where one material's latents live for a given quality, or nil if that
    /// pair was never trained.
    fileprivate static func ntcURL(dir: URL, material: String, quality: Quality) -> URL? {
        let url = dir.appendingPathComponent(quality.ntcFileName(base: material))
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    func selectQuality(index: Int) {
        guard ntcVariants.indices.contains(index), index != activeVariant else { return }
        activeVariant = index
        // the accumulated history is of the OLD latents; keep it and the resolve
        // blends the switch in over several frames instead of showing it
        historyValid = false
    }

    @objc func qualityChanged(_ sender: NSSegmentedControl) {
        selectQuality(index: sender.selectedSegment)
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


    fileprivate static func loadBlueNoise(device: any MTLDevice,
                                          url:    URL) throws -> any MTLTexture {
        let loader  = MTKTextureLoader(device: device)
        let texture = try loader.newTexture(URL: url, options: [
            .textureUsage:       NSNumber(value: MTLTextureUsage.shaderRead.rawValue),
            .textureStorageMode: NSNumber(value: MTLStorageMode.private.rawValue),
            .SRGB:               NSNumber(value: false),
        ])
        texture.label = "BlueNoise"
        print("blue noise \(texture.width)x\(texture.height) loaded")
        return texture
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
            // zero tangent (w = 0)
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
        lastDrawableSize = view.drawableSize

        let commandBuffer = queue.makeCommandBuffer()!
        var frame = UInt32(truncatingIfNeeded: frameIndex)
        var flags: UInt32 = stochasticLOD ? RenderFlags.stochasticLOD : 0

        if benchmark, benchmarkTensorOps, let benchmarkDecodePSO = benchmarkDecodePSO {
            var resource = ntcVariants[activeVariant].resources[0]
            let target = drawable.texture
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.setComputePipelineState(benchmarkDecodePSO)
            encoder.setTexture(resource.latentTexture, index: 0)
            encoder.setTexture(blueNoise!, index: 4)
            encoder.setTexture(target,     index: 9)
            encoder.setBuffer(resource.mlpBuffer,    offset: 0, index: 1)
            encoder.setBuffer(resource.constsBuffer, offset: 0, index: 2)
            encoder.setBytes(&resource.materialLayout,
                             length: MemoryLayout<MaterialLayout>.stride, index: 3)
            encoder.setBytes(&resource.gridDequant,
                             length: MemoryLayout<SIMD2<Float>>.stride,   index: 4)
            encoder.setBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setBytes(&flags, length: MemoryLayout<UInt32>.stride, index: 7)
            let side = Renderer.decodeTileSide
            encoder.dispatchThreadgroups(
                MTLSize(width:  (target.width  + side - 1) / side,
                        height: (target.height + side - 1) / side, depth: 1),
                threadsPerThreadgroup: MTLSize(width: Renderer.decodeTileThreads, height: 1, depth: 1))
            encoder.endEncoding()
        } else if benchmark {
            guard let renderPassDesc = view.currentRenderPassDescriptor else { return }
            let encoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDesc)!
            encoder.setRenderPipelineState(meshPSO)
            encoder.setDepthStencilState(depthState)
            var resource = ntcVariants[activeVariant].resources[0]
            encoder.setCullMode(.none)
            encoder.setFragmentTexture(resource.latentTexture, index: 0)
            encoder.setFragmentTexture(blueNoise!,             index: 4)
            encoder.setFragmentBuffer(resource.mlpBuffer,    offset: 0, index: 1)
            encoder.setFragmentBuffer(resource.constsBuffer, offset: 0, index: 2)
            encoder.setFragmentBytes(&resource.materialLayout,
                                     length: MemoryLayout<MaterialLayout>.stride, index: 3)
            encoder.setFragmentBytes(&resource.gridDequant,
                                     length: MemoryLayout<SIMD2<Float>>.stride,   index: 4)
            encoder.setFragmentBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 5)
            encoder.setFragmentBytes(&flags, length: MemoryLayout<UInt32>.stride, index: 7)
            encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            encoder.endEncoding()
        } else {
            guard let sceneColor = sceneColorTexture,
                  let sceneDepth = sceneDepthTexture,
                  let resolvePSO = resolvePSO,
                  historyTextures.count == 2 else { return }

            let camEye = SIMD3<Float>(0, 0.0, cameraDistance)
            if autoRotate {
                yaw = Float(CACurrentMediaTime() - startTime) * 0.7
            }
            // yaw then pitch, both pure rotations -- see the note on `yaw`
            let spin = matrix4x4_rotation(yaw, 0, 1, 0) * matrix4x4_rotation(pitch, 1, 0, 0)
            let viewMatrix = matrix_look_at_right_hand(camEye,
                                                       SIMD3<Float>(0, 0, 0),
                                                       SIMD3<Float>(0, 1, 0))
            let projMatrix = matrix_perspective_right_hand(radians_from_degrees(60),
                                                           aspect, 0.1, 100.0)
            let viewProj = projMatrix * viewMatrix

            var invViewProj = simd_inverse(viewProj)
            var camPos = camEye
            let variant = ntcVariants[activeVariant]

            if let gbufferUvLod = gbufferUvLod,
               let gbufferNormal = gbufferNormal,
               let gbufferTangent = gbufferTangent,
               let gbufferPSO = gbufferPSO,
               let decodePSO = decodePSO {

                // pass 1: geometry into the G-buffer
                let gbufferDesc = MTLRenderPassDescriptor()
                gbufferDesc.colorAttachments[0].texture     = gbufferUvLod
                gbufferDesc.colorAttachments[0].loadAction  = .clear
                // material index -1 marks a pixel with no geometry
                gbufferDesc.colorAttachments[0].clearColor  = MTLClearColor(red: 0, green: 0, blue: 0, alpha: -1)
                gbufferDesc.colorAttachments[0].storeAction = .store
                gbufferDesc.colorAttachments[1].texture     = gbufferNormal
                gbufferDesc.colorAttachments[1].loadAction  = .clear
                gbufferDesc.colorAttachments[1].clearColor  = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                gbufferDesc.colorAttachments[1].storeAction = .store
                gbufferDesc.colorAttachments[2].texture     = gbufferTangent
                gbufferDesc.colorAttachments[2].loadAction  = .clear
                gbufferDesc.colorAttachments[2].clearColor  = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                gbufferDesc.colorAttachments[2].storeAction = .store
                gbufferDesc.depthAttachment.texture     = sceneDepth
                gbufferDesc.depthAttachment.loadAction  = .clear
                gbufferDesc.depthAttachment.clearDepth  = 0.0
                // the decode pass reads this back for the world position
                gbufferDesc.depthAttachment.storeAction = .store

                let geometry = commandBuffer.makeRenderCommandEncoder(descriptor: gbufferDesc)!
                geometry.setRenderPipelineState(gbufferPSO)
                geometry.setDepthStencilState(depthState)
                geometry.setCullMode(.back)
                geometry.setFrontFacing(.counterClockwise)

                for submesh in submeshes {
                    let model = spin * recenter * submesh.transform
                    var uniforms = MeshUniforms(mvp: viewProj * model,
                                                model: model,
                                                normalMatrix: matrix_inverse_transpose(model))
                    let resource = ntcVariants[activeVariant].resources[submesh.ntcIndex]
                    var materialIndex = UInt32(submesh.ntcIndex)

                    geometry.setVertexBuffer(submesh.vertexBuffer, offset: 0, index: 0)
                    geometry.setVertexBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 1)
                    geometry.setFragmentTexture(blueNoise!, index: 4)
                    geometry.setFragmentBuffer(resource.constsBuffer, offset: 0, index: 2)
                    geometry.setFragmentBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 5)
                    geometry.setFragmentBytes(&flags, length: MemoryLayout<UInt32>.stride, index: 7)
                    geometry.setFragmentBytes(&materialIndex, length: MemoryLayout<UInt32>.stride, index: 8)
                    geometry.drawIndexedPrimitives(type: .triangle,
                                                   indexCount: submesh.indexCount,
                                                   indexType: .uint32,
                                                   indexBuffer: submesh.indexBuffer,
                                                   indexBufferOffset: 0)
                }
                geometry.endEncoding()

                // pass 2: skybox fills the background; depth is loaded from pass 1
                // and tested equal, so it survives only where no geometry drew
                let skyDesc = MTLRenderPassDescriptor()
                skyDesc.colorAttachments[0].texture     = sceneColor
                skyDesc.colorAttachments[0].loadAction  = .clear
                skyDesc.colorAttachments[0].clearColor  = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
                skyDesc.colorAttachments[0].storeAction = .store
                skyDesc.depthAttachment.texture     = sceneDepth
                skyDesc.depthAttachment.loadAction  = .load
                skyDesc.depthAttachment.storeAction = .store

                let sky = commandBuffer.makeRenderCommandEncoder(descriptor: skyDesc)!
                sky.setRenderPipelineState(skyboxPSO!)
                sky.setDepthStencilState(skyboxDepthState!)
                sky.setCullMode(.none)
                sky.setFragmentTexture(environmentCubemap!, index: 0)
                sky.setFragmentBytes(&invViewProj, length: MemoryLayout<simd_float4x4>.stride, index: 0)
                sky.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                sky.endEncoding()

                // pass 3: tensor ops NTC decode + shading, ONE dispatch. Each
                // threadgroup loops over just the materials its own tile uses,
                // so the cost follows screen coverage, not the material count.
                let decode = commandBuffer.makeComputeCommandEncoder()!
                decode.setComputePipelineState(decodePSO)
                let tileSide = Renderer.decodeTileSide
                let tileCount = MTLSize(width:  (sceneColor.width  + tileSide - 1) / tileSide,
                                        height: (sceneColor.height + tileSide - 1) / tileSide,
                                        depth: 1)
                let tileThreads = MTLSize(width: Renderer.decodeTileThreads, height: 1, depth: 1)

                decode.setTexture(gbufferUvLod,   index: 5)
                decode.setTexture(gbufferNormal,  index: 6)
                decode.setTexture(gbufferTangent, index: 7)
                decode.setTexture(sceneDepth,     index: 8)
                decode.setTexture(sceneColor,     index: 9)
                decode.setTexture(irradianceMap!, index: 1)
                decode.setTexture(radianceMap!,   index: 2)
                decode.setTexture(brdfLut!,       index: 3)
                decode.setBuffer(variant.slots, offset: 0, index: 0)
                decode.setBytes(&camPos,      length: MemoryLayout<SIMD3<Float>>.stride, index: 6)
                decode.setBytes(&invViewProj, length: MemoryLayout<simd_float4x4>.stride, index: 9)

                // the slot buffer reaches these by address, so they need
                // explicit residency
                for resource in variant.resources {
                    decode.useResource(resource.latentTexture, usage: .read)
                    decode.useResource(resource.mlpBuffer,     usage: .read)
                    decode.useResource(resource.constsBuffer,  usage: .read)
                }

                decode.dispatchThreadgroups(tileCount, threadsPerThreadgroup: tileThreads)
                decode.endEncoding()
            } else {
                // no tensor ops: one MLP per fragment
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

                for submesh in submeshes {
                    let model = spin * recenter * submesh.transform
                    var uniforms = MeshUniforms(mvp: viewProj * model,
                                                model: model,
                                                normalMatrix: matrix_inverse_transpose(model))
                    var resource = ntcVariants[activeVariant].resources[submesh.ntcIndex] // value copy; setFragmentBytes needs inout

                    encoder.setVertexBuffer(submesh.vertexBuffer, offset: 0, index: 0)
                    encoder.setVertexBytes(&uniforms, length: MemoryLayout<MeshUniforms>.stride, index: 1)
                    encoder.setFragmentTexture(resource.latentTexture, index: 0)
                    encoder.setFragmentTexture(irradianceMap!, index: 1)
                    encoder.setFragmentTexture(radianceMap!,   index: 2)
                    encoder.setFragmentTexture(brdfLut!,       index: 3)
                    encoder.setFragmentTexture(blueNoise!,     index: 4)
                    encoder.setFragmentBuffer(resource.mlpBuffer,    offset: 0, index: 1)
                    encoder.setFragmentBuffer(resource.constsBuffer, offset: 0, index: 2)
                    encoder.setFragmentBytes(&resource.materialLayout,
                                             length: MemoryLayout<MaterialLayout>.stride, index: 3)
                    encoder.setFragmentBytes(&resource.gridDequant,
                                             length: MemoryLayout<SIMD2<Float>>.stride, index: 4)
                    encoder.setFragmentBytes(&frame, length: MemoryLayout<UInt32>.stride, index: 5)
                    encoder.setFragmentBytes(&camPos, length: MemoryLayout<SIMD3<Float>>.stride, index: 6)
                    encoder.setFragmentBytes(&flags, length: MemoryLayout<UInt32>.stride, index: 7)
                    encoder.drawIndexedPrimitives(type: .triangle,
                                                  indexCount: submesh.indexCount,
                                                  indexType: .uint32,
                                                  indexBuffer: submesh.indexBuffer,
                                                  indexBufferOffset: 0)
                }

                encoder.setRenderPipelineState(skyboxPSO!)
                encoder.setDepthStencilState(skyboxDepthState!)
                encoder.setCullMode(.none)
                encoder.setFragmentTexture(environmentCubemap!, index: 0)
                encoder.setFragmentBytes(&invViewProj, length: MemoryLayout<simd_float4x4>.stride, index: 0)
                encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
                encoder.endEncoding()
            }

            // pass 2: temporal resolve
            let readIdx  = historyIndex
            let writeIdx = 1 - historyIndex
            var uniforms = TemporalUniforms(viewProj:      viewProj,
                                            invViewProj:   invViewProj,
                                            deltaRotation: prevSpin * simd_inverse(spin),
                                            historyValid:  (historyValid && temporalAccumulation) ? 1 : 0)

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
        precondition(ntc.mlp.count == mlpCount, "mlp half count \(ntc.mlp.count) != expected \(mlpCount)")

        let latentTex = buildLatentTexture(device: device,
                                           grid: ntc.grid,
                                           pyramidSizes: ntc.pyramidSizes.map { Int($0) },
                                           pyramidOffsets: pyramidOffsets,
                                           fPerGrid: fPerGrid)

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
    /// Storage is `abgr4Unorm`, 4 codes packed into one 16-bit texel, which is
    /// filterable on Apple GPUs. The grid is 4-bit everywhere (see BITS in the
    /// trainer); readNTC refuses a .ntc that says otherwise.
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
                                           fPerGrid:       Int) -> any MTLTexture {
        let kGrids = pyramidSizes.count
        precondition(fPerGrid % 4 == 0, "fPerGrid \(fPerGrid) must be a multiple of 4")
        let slices   = fPerGrid / 4
        let baseSize = pyramidSizes[0]

        let texDesc = MTLTextureDescriptor()
        texDesc.textureType      = .type2DArray
        texDesc.pixelFormat      = .abgr4Unorm
        texDesc.width            = baseSize
        texDesc.height           = baseSize
        texDesc.arrayLength      = slices
        texDesc.mipmapLevelCount = kGrids
        texDesc.usage            = .shaderRead
        texDesc.storageMode      = .shared
        let texture = device.makeTexture(descriptor: texDesc)!
        texture.label = "NTC.latents (4-bit)"

        for mip in 0..<kGrids {
            let width = pyramidSizes[mip]
            precondition(width == max(baseSize >> mip, 1), "grid \(mip) size \(width) is not mip \(mip) of \(baseSize); pyramid must be power-of-two")
            let gridBase = pyramidOffsets[mip]
            for slice in 0..<slices {
                var texels = [UInt16](repeating: 0, count: width * width)
                for y in 0..<width {
                    for x in 0..<width {
                        let srcIndex = (y * width + x) * fPerGrid + gridBase + slice * 4
                        // abgr4Unorm packs the four codes into one 16-bit
                        // texel, .r in the most significant 4 bits and .a in the least
                        let codeR = UInt16(grid[srcIndex + 0] & 0x0F)   // bits [12,16)
                        let codeG = UInt16(grid[srcIndex + 1] & 0x0F)   // bits [8,12)
                        let codeB = UInt16(grid[srcIndex + 2] & 0x0F)   // bits [4,8)
                        let codeA = UInt16(grid[srcIndex + 3] & 0x0F)   // bits [0,4)
                        texels[y * width + x] = (codeR << 12) | (codeG << 8) | (codeB << 4) | codeA
                    }
                }
                texels.withUnsafeBytes { raw in
                    texture.replace(region: MTLRegionMake2D(0, 0, width, width),
                                    mipmapLevel: mip, slice: slice,
                                    withBytes: raw.baseAddress!,
                                    bytesPerRow: width * 2, bytesPerImage: width * width * 2)
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
