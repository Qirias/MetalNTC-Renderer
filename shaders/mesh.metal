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

/// Where each semantic lives in the decoded NTC output. -1 means the .ntc does
/// not carry that semantic, so the sampler substitutes a neutral default.
/// Built on the host from the .ntc slot table, so trimming the manifest down to
/// a single texture does not require a shader change.
struct MaterialLayout {
    int albedo;
    int normal;
    int roughness;
    int metalness;
    int occlusion;
    int emissive;
};

/// One decoded surface point. Every semantic has a field whether or not this
/// .ntc carries it -- shading code can read them all today, and only the ones
/// actually trained carry real data.
struct Material {
    float3 albedo;
    float3 normal;      // tangent space, unpacked to [-1, 1]
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

struct BenchOut {
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

vertex BenchOut fullscreen_vs(uint vertexID [[vertex_id]]) {
    float2 uv = float2(float((vertexID << 1) & 2), float(vertexID & 2));
    BenchOut output;
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

constant bool kStochasticFilter = true;

static inline uint select_lod(float2 uv, float2 pixelCoord, uint frameIndex,
                              constant StepConstants& consts) {
    float2 dx = dfdx(uv) * float(consts.srcW);
    float2 dy = dfdy(uv) * float(consts.srcH);
    float lodf = clamp(0.5 * log2(max(dot(dx, dx), dot(dy, dy))),
                       0.0, float(consts.mipCount - 1));

    if (kStochasticFilter) {
        float base = floor(lodf);
        float frac = lodf - base;
        float jitter = hash01(uint(pixelCoord.x), uint(pixelCoord.y), frameIndex);
        // choose mip stochasticly but use frac as the probability
        return min(uint(base) + (jitter < frac ? 1u : 0u), consts.mipCount - 1u);
    }
    return uint(round(lodf));
}

static Material sample_material(float2                   uv,
                                uint                     lod,
                                texture2d_array<float>   latents,
                                float2                   gridDequant,
                                device const half*       mlp,
                                constant StepConstants&  consts,
                                constant MaterialLayout& layout) {
    half pred[K_OUT_MAX];
    ntc_decode_quant(uv, lod, latents, latentSampler,
                     gridDequant.x, gridDequant.y, mlp, consts, pred);

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

// PBR
fragment float4 mesh_fs(VertexOut                  in          [[stage_in]],
                        texture2d_array<float>     latents     [[texture(0)]],
                        texturecube<float>         irradianceMap [[texture(1)]],
                        texturecube<float>         radianceMap   [[texture(2)]],
                        texture2d<float>           brdfLut       [[texture(3)]],
                        device const half*         mlp         [[buffer(1)]],
                        constant StepConstants&    consts      [[buffer(2)]],
                        constant MaterialLayout&   layout      [[buffer(3)]],
                        constant float2&           gridDequant [[buffer(4)]],
                        constant uint&             frameIndex  [[buffer(5)]],
                        constant float3&           cameraPos   [[buffer(6)]]) {
    uint lod = select_lod(in.uv, in.position.xy, frameIndex, consts);
    Material material = sample_material(in.uv, lod, latents, gridDequant, mlp, consts, layout);

    float3 geometricNormal = normalize(in.worldNormal);
    float3 shadingNormal;
    if (abs(in.tangentSign) < 0.5 || length(in.worldTangent) < 1e-4) {
        shadingNormal = geometricNormal;
    } else {
        float3 tangent   = normalize(in.worldTangent - geometricNormal * dot(geometricNormal, in.worldTangent));
        float3 bitangent = cross(geometricNormal, tangent) * in.tangentSign;
        shadingNormal    = normalize(float3x3(tangent, bitangent, geometricNormal) * material.normal);
    }

    float3 albedo    = srgbToLinear(material.albedo);
    float3 emissive  = srgbToLinear(material.emissive);
    float  roughness = clamp(material.roughness, 0.045, 1.0);
    float  metalness = clamp(material.metalness, 0.0, 1.0);

    float3 viewDir    = normalize(cameraPos - in.worldPos);
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
    float3 color   = ambient + emissive;

    // output linear HDR into the offscreen buffer; the temporal resolve pass
    // tone-maps
    return float4(color, in.position.z);
}

fragment float4 bench_fs(BenchOut                   in          [[stage_in]],
                         texture2d_array<float>     latents     [[texture(0)]],
                         device const half*         mlp         [[buffer(1)]],
                         constant StepConstants&    consts      [[buffer(2)]],
                         constant MaterialLayout&   layout      [[buffer(3)]],
                         constant float2&           gridDequant [[buffer(4)]],
                         constant uint&             frameIndex  [[buffer(5)]]) {
    uint lod = select_lod(in.uv, in.position.xy, frameIndex, consts);
    Material material = sample_material(in.uv, lod, latents, gridDequant, mlp, consts, layout);
    float3 color = material.albedo + material.emissive;
    return float4(clamp(color, 0.0, 1.0), 1.0);
}
