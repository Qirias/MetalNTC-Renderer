// MetalNTC Renderer — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

#include <metal_stdlib>
#include "../../NTCCore/shaders/common.h"
#include "pbr_common.h"
using namespace metal;

struct MeshVertex {
    packed_float3 position;
    packed_float3 normal;
    packed_float2 uv;
    packed_float4 tangent;
};

struct MeshUniforms {
    float4x4 mvp;
    float4x4 model;
    float4x4 normalMatrix;
};

struct MaterialLayout {
    int albedo;
    int normal;
    int roughness;
    int metalness;
    int occlusion;
    int emissive;
};

struct Material {
    float3 albedo;
    float3 normal;
    float  roughness;
    float  metalness;
    float  occlusion;
    float3 emissive;
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 worldNormal;
    float3 worldTangent;
    float  tangentSign;
    float2 uv;
};

struct BenchmarkOut {
    float4 position [[position]];
    float2 uv;
};

vertex VertexOut mesh_vs(uint                      vertexID [[vertex_id]],
                    device const MeshVertex*  vertices [[buffer(0)]],
                    constant MeshUniforms&    uniforms [[buffer(1)]]) {
    MeshVertex vertexIn = vertices[vertexID];
    VertexOut output;
    output.position     = uniforms.mvp * float4(float3(vertexIn.position), 1.0);
    output.worldPos     = (uniforms.model * float4(float3(vertexIn.position), 1.0)).xyz;
    output.worldNormal  = (uniforms.normalMatrix * float4(float3(vertexIn.normal), 0.0)).xyz;
    output.worldTangent = (uniforms.model * float4(float3(vertexIn.tangent.xyz), 0.0)).xyz;
    output.tangentSign  = vertexIn.tangent.w;
    output.uv           = float2(vertexIn.uv);
    return output;
}

vertex BenchmarkOut fullscreen_vs(uint vertexID [[vertex_id]]) {
    float2 uv = float2(float((vertexID << 1) & 2), float(vertexID & 2));
    BenchmarkOut output;
    output.position = float4(uv * 2.0 - 1.0, 1.0, 1.0);
    output.uv       = uv;
    return output;
}

static inline float3 read3(thread const half* pred, int base, float3 fallback) {
    return base < 0 ? fallback
                    : float3(pred[base], pred[base + 1], pred[base + 2]);
}

static inline float read1(thread const half* pred, int base, float fallback) {
    return base < 0 ? fallback : float(pred[base]);
}

constexpr sampler latentSampler(coord::normalized,
                                address::clamp_to_edge,
                                filter::linear,
                                mip_filter::linear);

constexpr sampler iblSampler(filter::linear, mip_filter::linear, address::clamp_to_edge);
constexpr sampler brdfSampler(filter::linear, address::clamp_to_edge);

#define RENDER_FLAG_STOCHASTIC_LOD  (1u << 0)

static inline uint quantize_lod(float lodf, float2 pixelCoord, uint frameIndex,
                                bool stochastic, texture2d<float> blueNoise,
                                constant StepConstants& consts) {
    if (stochastic) {
        float base = floor(lodf);
        float frac = lodf - base;

        uint bnW = blueNoise.get_width();
        uint bnH = blueNoise.get_height();
        float bn = blueNoise.read(uint2(uint(pixelCoord.x) % bnW,
                                        uint(pixelCoord.y) % bnH)).r;
        // advancing by the golden ratio per frame
        float jitter = fract(bn + float(frameIndex) * 0.6180339887f);
        // choose mip stochasticly but use frac as the probability
        return min(uint(base) + (jitter < frac ? 1u : 0u), consts.mipCount - 1u);
    }
    return uint(round(lodf));
}

static inline uint select_lod(float2 uv, float2 pixelCoord, uint frameIndex,
                              bool stochastic, texture2d<float> blueNoise,
                              constant StepConstants& consts) {
    float2 dx = dfdx(uv) * float(consts.srcW);
    float2 dy = dfdy(uv) * float(consts.srcH);
    float lodf = clamp(0.5 * log2(max(dot(dx, dx), dot(dy, dy))),
                       0.0, float(consts.mipCount - 1));
    return quantize_lod(lodf, pixelCoord, frameIndex, stochastic, blueNoise, consts);
}

static Material unpack_material(thread const half* pred, MaterialLayout layout) {
    Material m;
    m.albedo    = read3(pred, layout.albedo,    float3(0.5));
    m.roughness = read1(pred, layout.roughness, 1.0);
    m.metalness = read1(pred, layout.metalness, 0.0);
    m.occlusion = read1(pred, layout.occlusion, 1.0);
    m.emissive  = read3(pred, layout.emissive,  float3(0.0));

    float3 packedN = read3(pred, layout.normal, float3(0.5, 0.5, 1.0));
    m.normal = packedN * 2.0 - 1.0;

    return m;
}

static Material sample_material(float2                   uv,
                                uint                     lod,
                                texture2d_array<float>   latents,
                                float2                   gridDequant,
                                device const half*       mlp,
                                constant StepConstants&  consts,
                                constant MaterialLayout& layout) {
    half pred[K_OUT_MAX];
    // wrap here, not before select_lod: the lod needs the UNWRAPPED derivatives,
    // but the latent sampler is clamp_to_edge, so a tiled uv would pin every
    // pixel to the texture border
    ntc_decode_quant(fract(uv), lod, latents, latentSampler,
                     gridDequant.x, gridDequant.y, mlp, consts, pred);
    return unpack_material(pred, layout);
}


static float3 shade_material(Material           material,
                             float3             worldPos,
                             float3             worldNormal,
                             float3             worldTangent,
                             float              tangentSign,
                             float3             cameraPos,
                             texturecube<float> irradianceMap,
                             texturecube<float> radianceMap,
                             texture2d<float>   brdfLut) {
    float3 geometricNormal = normalize(worldNormal);
    float3 shadingNormal;
    if (abs(tangentSign) < 0.5 || length(worldTangent) < 1e-4) {
        shadingNormal = geometricNormal;
    } else {
        float3 tangent   = normalize(worldTangent - geometricNormal * dot(geometricNormal, worldTangent));
        float3 bitangent = cross(geometricNormal, tangent) * tangentSign;
        shadingNormal    = normalize(float3x3(tangent, bitangent, geometricNormal) * material.normal);
    }

    float3 albedo    = srgbToLinear(material.albedo);
    float3 emissive  = srgbToLinear(material.emissive);
    float  roughness = clamp(material.roughness, 0.045, 1.0);
    float  metalness = clamp(material.metalness, 0.0, 1.0);

    float3 viewDir    = normalize(cameraPos - worldPos);
    float3 reflectDir = reflect(-viewDir, shadingNormal);
    float  NdotV      = max(dot(shadingNormal, viewDir), 0.0);

    float3 baseReflectance = mix(float3(0.04), albedo, metalness);
    float3 fresnel         = fresnelSchlickRoughness(NdotV, baseReflectance, roughness);
    float3 diffuseWeight   = (1.0 - fresnel) * (1.0 - metalness);

    // diffuse
    float3 irradiance = irradianceMap.sample(iblSampler, shadingNormal).rgb;
    float3 diffuse    = irradiance * albedo;

    // specular
    const float MAX_REFLECTION_LOD = 9.0;
    float3 prefiltered = radianceMap.sample(iblSampler, reflectDir, level(roughness * roughness * MAX_REFLECTION_LOD)).rgb;
    float2 brdf        = brdfLut.sample(brdfSampler, float2(NdotV, roughness)).rg;
    float3 specular    = prefiltered * (fresnel * brdf.x + brdf.y);

    float3 ambient = (diffuseWeight * diffuse + specular) * material.occlusion;
    return ambient + emissive;
}

fragment float4 mesh_fs(VertexOut                  in          [[stage_in]],
                        texture2d_array<float>     latents     [[texture(0)]],
                        texturecube<float>         irradianceMap [[texture(1)]],
                        texturecube<float>         radianceMap   [[texture(2)]],
                        texture2d<float>           brdfLut       [[texture(3)]],
                        texture2d<float>           blueNoise     [[texture(4)]],
                        device const half*         mlp         [[buffer(1)]],
                        constant StepConstants&    consts      [[buffer(2)]],
                        constant MaterialLayout&   layout      [[buffer(3)]],
                        constant float2&           gridDequant [[buffer(4)]],
                        constant uint&             frameIndex  [[buffer(5)]],
                        constant float3&           cameraPos   [[buffer(6)]],
                        constant uint&             renderFlags [[buffer(7)]]) {
    uint lod = select_lod(in.uv, in.position.xy, frameIndex,
                          (renderFlags & RENDER_FLAG_STOCHASTIC_LOD) != 0u,
                          blueNoise, consts);
    Material material = sample_material(in.uv, lod, latents, gridDequant, mlp, consts, layout);

    float3 color = shade_material(material, in.worldPos, in.worldNormal,
                                  in.worldTangent, in.tangentSign, cameraPos,
                                  irradianceMap, radianceMap, brdfLut);

    // output linear HDR into the offscreen buffer; the temporal resolve pass tone-maps
    return float4(color, in.position.z);
}

fragment float4 benchmark_fs(BenchmarkOut                   in          [[stage_in]],
                         texture2d_array<float>     latents     [[texture(0)]],
                         texture2d<float>           blueNoise   [[texture(4)]],
                         device const half*         mlp         [[buffer(1)]],
                         constant StepConstants&    consts      [[buffer(2)]],
                         constant MaterialLayout&   layout      [[buffer(3)]],
                         constant float2&           gridDequant [[buffer(4)]],
                         constant uint&             frameIndex  [[buffer(5)]],
                         constant uint&             renderFlags [[buffer(7)]]) {   // mesh_fs has cameraPos at 6
    uint lod = select_lod(in.uv, in.position.xy, frameIndex,
                          (renderFlags & RENDER_FLAG_STOCHASTIC_LOD) != 0u,
                          blueNoise, consts);
    Material material = sample_material(in.uv, lod, latents, gridDequant, mlp, consts, layout);
    float3 color = material.albedo + material.emissive;
    return float4(clamp(color, 0.0, 1.0), 1.0);
}

#ifdef NTC_TENSOR_OPS

// ne neural material, reachable without rebinding, so a single dispatch can
// decode any of them and each threadgroup loops over just the ones its own tile uses
struct NTCMaterialSlot {
    texture2d_array<float>      latents;
    device const half*          mlp;
    device const StepConstants* consts;
    MaterialLayout              layout;
    float2                      gridDequant;
};

// no neural material at this pixel
#define NTC_NO_MATERIAL 0x7FFFFFFFu

struct GBufferOut {
    float4 uvLodMaterial [[color(0)]];  // uv.xy, lod, material index (-1 = no geometry)
    float4 normalSign    [[color(1)]];  // world normal.xyz, tangent sign
    float4 tangent       [[color(2)]];  // world tangent.xyz, unused
};

fragment GBufferOut gbuffer_fs(VertexOut               in            [[stage_in]],
                               texture2d<float>        blueNoise     [[texture(4)]],
                               constant StepConstants& consts        [[buffer(2)]],
                               constant uint&          frameIndex    [[buffer(5)]],
                               constant uint&          renderFlags   [[buffer(7)]],
                               constant uint&          materialIndex [[buffer(8)]]) {
    // select_lod needs screen-space derivatives, it can't be computed
    // efficiently in the compute pass
    uint lod = select_lod(in.uv, in.position.xy, frameIndex,
                          (renderFlags & RENDER_FLAG_STOCHASTIC_LOD) != 0u,
                          blueNoise, consts);

    GBufferOut out;
    // fract for the same reason as sample_material; lod above already used the
    // unwrapped derivatives
    out.uvLodMaterial = float4(fract(in.uv), float(lod), float(materialIndex));
    out.normalSign    = float4(in.worldNormal, in.tangentSign);
    out.tangent       = float4(in.worldTangent, 0.0);
    return out;
}

// dispatching 128 threads for 64 pixels means there are idle threads;
// but execution_simdgroups<4> means four simdgroups and all 128 lanes must reach
// the matmul, because the cooperative tensor is distributed across all their
// registers; execution_simdgroups<2> would give a 1:1 mapping and measured 2% SLOWER
#define DECODE_TILE_SIDE 8

static inline float3 world_position_from_depth(float2 ndc, float depth, float4x4 invViewProj) {
    float4 world = invViewProj * float4(ndc, depth, 1.0);
    return world.xyz / world.w;
}

kernel void ntc_decode_shade(texture2d<float>                gbufferUvLod   [[texture(5)]],
                             texture2d<float>                gbufferNormal  [[texture(6)]],
                             texture2d<float>                gbufferTangent [[texture(7)]],
                             depth2d<float>                  gbufferDepth   [[texture(8)]],
                             texture2d<float, access::write> sceneColor     [[texture(9)]],
                             texturecube<float>              irradianceMap  [[texture(1)]],
                             texturecube<float>              radianceMap    [[texture(2)]],
                             texture2d<float>                brdfLut        [[texture(3)]],
                             device const NTCMaterialSlot*   slots          [[buffer(0)]],
                             constant float3&                cameraPos      [[buffer(6)]],
                             constant float4x4&              invViewProj    [[buffer(9)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint  tid  [[thread_index_in_threadgroup]]) {
    threadgroup half features[TILE_SIZE * F_IN];
    threadgroup half hidden  [TILE_SIZE * K_HIDDEN];
    threadgroup half pred    [TILE_SIZE * K_OUT_MAX];
    threadgroup atomic_uint nextMaterial;

    uint2 screenSize = uint2(gbufferNormal.get_width(), gbufferNormal.get_height());
    uint2 pixel = tgid * DECODE_TILE_SIDE + uint2(tid % DECODE_TILE_SIDE, tid / DECODE_TILE_SIDE);
    // tid < TILE_SIZE has no pixel but will still be used by the cooperative tensor
    bool hasPixel = tid < TILE_SIZE && pixel.x < screenSize.x && pixel.y < screenSize.y;

    float4 uvLodMaterial = hasPixel ? gbufferUvLod.read(pixel) : float4(0, 0, 0, -1);
    bool pending    = uvLodMaterial.w >= 0.0;
    uint myMaterial = pending ? uint(uvLodMaterial.w) : NTC_NO_MATERIAL;

    // one pass per distinct material in this tile; each retires at least one of
    // them, so this terminates
    while (true) {
        if (tid == 0) {
            atomic_store_explicit(&nextMaterial, NTC_NO_MATERIAL, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (pending) {
            atomic_fetch_min_explicit(&nextMaterial, myMaterial, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // uniform across the threadgroup, which it must be: every lane has to
        // reach mlp_forward_tensor_ops
        uint materialIndex = atomic_load_explicit(&nextMaterial, memory_order_relaxed);
        if (materialIndex == NTC_NO_MATERIAL) break;

        device const NTCMaterialSlot& slot = slots[materialIndex];

        // every row is filled, even ones belonging to another material; their
        // results are simply not written out
        if (tid < TILE_SIZE) {
            half row[F_IN];
            ntc_build_features(uvLodMaterial.xy, uint(max(uvLodMaterial.z, 0.0)),
                               slot.latents, latentSampler,
                               slot.gridDequant.x, slot.gridDequant.y, *slot.consts, row);
            for (uint i = 0; i < F_IN; i++) {
                features[tid * F_IN + i] = row[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        mlp_forward_tensor_ops(slot.mlp,
                               slot.consts->offsetW1, slot.consts->offsetB1,
                               slot.consts->offsetW2, slot.consts->offsetB2,
                               slot.consts->offsetW3, slot.consts->offsetB3,
                               features, hidden, pred);

        if (pending && myMaterial == materialIndex) {
            half row[K_OUT_MAX];
            for (uint k = 0; k < K_OUT_MAX; k++) {
                row[k] = pred[tid * K_OUT_MAX + k];
            }
            Material material = unpack_material(row, slot.layout);

            float  depth = gbufferDepth.read(pixel);
            float2 uv    = (float2(pixel) + 0.5) / float2(screenSize);
            float2 ndc   = float2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
            float3 worldPos = world_position_from_depth(ndc, depth, invViewProj);

            float4 normalSign = gbufferNormal.read(pixel);
            float4 tangent    = gbufferTangent.read(pixel);

            float3 color = shade_material(material, worldPos, normalSign.xyz,
                                          tangent.xyz, normalSign.w, cameraPos,
                                          irradianceMap, radianceMap, brdfLut);

            // same convention as mesh_fs: linear HDR, depth in alpha
            sceneColor.write(float4(color, depth), pixel);
            pending = false;
        }
    }
}

kernel void benchmark_decode(texture2d_array<float>          latents     [[texture(0)]],
                         texture2d<float>                blueNoise   [[texture(4)]],
                         texture2d<float, access::write> output      [[texture(9)]],
                         device const half*              mlp         [[buffer(1)]],
                         constant StepConstants&         consts      [[buffer(2)]],
                         constant MaterialLayout&        layout      [[buffer(3)]],
                         constant float2&                gridDequant [[buffer(4)]],
                         constant uint&                  frameIndex  [[buffer(5)]],
                         constant uint&                  renderFlags [[buffer(7)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint  tid  [[thread_index_in_threadgroup]]) {
    threadgroup half features[TILE_SIZE * F_IN];
    threadgroup half hidden  [TILE_SIZE * K_HIDDEN];
    threadgroup half pred    [TILE_SIZE * K_OUT_MAX];

    uint2 screenSize = uint2(output.get_width(), output.get_height());
    uint2 pixel = tgid * DECODE_TILE_SIDE + uint2(tid % DECODE_TILE_SIDE, tid / DECODE_TILE_SIDE);
    bool  hasPixel = tid < TILE_SIZE && pixel.x < screenSize.x && pixel.y < screenSize.y;

    // benchmark_fs gets this from dfdx/dfdy on a full-screen quad whose uv spans
    // [0,1]; that reduces to a constant, so derive it directly
    float lodf = clamp(log2(max(float(consts.srcW) / float(screenSize.x),
                                float(consts.srcH) / float(screenSize.y))),
                       0.0, float(consts.mipCount - 1));

    float2 uv = (float2(pixel) + 0.5) / float2(screenSize);
    if (tid < TILE_SIZE) {
        uint lod = quantize_lod(lodf, float2(pixel), frameIndex,
                                (renderFlags & RENDER_FLAG_STOCHASTIC_LOD) != 0u,
                                blueNoise, consts);
        half row[F_IN];
        ntc_build_features(uv, lod, latents, latentSampler,
                           gridDequant.x, gridDequant.y, consts, row);
        for (uint i = 0; i < F_IN; i++) {
            features[tid * F_IN + i] = row[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    mlp_forward_tensor_ops(mlp,
                        consts.offsetW1, consts.offsetB1,
                        consts.offsetW2, consts.offsetB2,
                        consts.offsetW3, consts.offsetB3,
                        features, hidden, pred);

    if (!hasPixel) {
        return;
    }

    half row[K_OUT_MAX];
    for (uint k = 0; k < K_OUT_MAX; k++) {
        row[k] = pred[tid * K_OUT_MAX + k];
    }
    Material material = unpack_material(row, layout);
    float3 color = material.albedo + material.emissive;
    output.write(float4(clamp(color, 0.0, 1.0), 1.0), pixel);
}

#endif  // NTC_TENSOR_OPS
