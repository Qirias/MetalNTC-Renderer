// MetalNTC Renderer — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

#include <metal_stdlib>
using namespace metal;

constexpr sampler linearSampler(mag_filter::linear, min_filter::linear);
constexpr sampler linearRepeatSampler(mag_filter::linear, min_filter::linear, address::repeat);

static inline float3 cubemapDirectionFromCoords(uint face, float2 uv) {
    float2 ndc = uv * 2.0 - 1.0;
    float3 direction;
    switch (face) {
        case 0: direction = float3( 1.0, -ndc.y, -ndc.x); break; // +X
        case 1: direction = float3(-1.0, -ndc.y,  ndc.x); break; // -X
        case 2: direction = float3( ndc.x,  1.0,  ndc.y); break; // +Y
        case 3: direction = float3( ndc.x, -1.0, -ndc.y); break; // -Y
        case 4: direction = float3( ndc.x, -ndc.y,  1.0); break; // +Z
        default:direction = float3(-ndc.x, -ndc.y, -1.0); break; // -Z
    }
    return normalize(direction);
}

static inline float2 directionToEquirectangularUV(float3 direction) {
    float phi   = atan2(direction.z, direction.x + 1e-6);
    float theta = asin(clamp(direction.y, -1.0, 1.0));
    float longitude = (phi + M_PI_F) / (2.0 * M_PI_F);
    float latitude  = 1.0 - (theta + M_PI_2_F) / M_PI_F;   // flip so row 0 = top
    return float2(longitude, latitude);
}

kernel void equirect_to_cubemap(texture2d<float, access::sample>     equirect [[texture(0)]],
                                texturecube<float, access::write>    cubemap  [[texture(1)]],
                                uint3 threadPos [[thread_position_in_grid]]) {
    uint width = cubemap.get_width(), height = cubemap.get_height();
    if (threadPos.x >= width || threadPos.y >= height || threadPos.z >= 6) return;

    float2 uv        = float2(threadPos.xy) / float2(width, height);
    float3 direction = cubemapDirectionFromCoords(threadPos.z, uv);
    float4 color     = equirect.sample(linearRepeatSampler, directionToEquirectangularUV(direction));
    cubemap.write(color, threadPos.xy, threadPos.z);
}

kernel void convolve_irradiance(texturecube<float, access::sample>  environment [[texture(0)]],
                                texturecube<float, access::write>   irradianceMap [[texture(1)]],
                                uint3 threadPos [[thread_position_in_grid]]) {
    uint width = irradianceMap.get_width(), height = irradianceMap.get_height();
    if (threadPos.x >= width || threadPos.y >= height || threadPos.z >= 6) return;

    float2 uv     = float2(threadPos.xy) / float2(width, height);
    float3 normal = cubemapDirectionFromCoords(threadPos.z, uv);

    float3 up    = float3(0.0, 1.0, 0.0);
    float3 right = normalize(cross(up, normal));
    up           = normalize(cross(normal, right));

    float3 irradiance  = float3(0.0);
    float  sampleCount = 0.0;
    const float sampleDelta = 0.025;
    for (float phi = 0.0; phi < 2.0 * M_PI_F; phi += sampleDelta) {
        for (float theta = 0.0; theta < 0.5 * M_PI_F; theta += sampleDelta) {
            float3 tangentSample = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
            float3 sampleDir = tangentSample.x * right + tangentSample.y * up + tangentSample.z * normal;
            irradiance  += environment.sample(linearSampler, sampleDir).rgb * cos(theta) * sin(theta);
            sampleCount += 1.0;
        }
    }
    irradiance = M_PI_F * irradiance / sampleCount;
    irradianceMap.write(float4(irradiance, 1.0), threadPos.xy, threadPos.z);
}

static float radicalInverseVdC(uint bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10;
}

static float2 hammersley(uint index, uint count) {
    return float2(float(index) / float(count), radicalInverseVdC(index));
}

static float3 importanceSampleGGX(float2 randomSample, float3 normal, float roughness) {
    float alpha = roughness * roughness;
    float phi = 2.0 * M_PI_F * randomSample.x;
    float cosTheta = sqrt((1.0 - randomSample.y) / (1.0 + (alpha * alpha - 1.0) * randomSample.y));
    float sinTheta = sqrt(1.0 - cosTheta * cosTheta);
    float3 halfTangent = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    float3 up        = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(1.0, 0.0, 0.0);
    float3 tangent   = normalize(cross(up, normal));
    float3 bitangent = cross(normal, tangent);
    return normalize(tangent * halfTangent.x + bitangent * halfTangent.y + normal * halfTangent.z);
}

static float geometrySchlickGGX(float nDotV, float roughness) {
    float k = (roughness * roughness) / 2.0;
    return nDotV / (nDotV * (1.0 - k) + k);
}

static float geometrySmith(float3 normal, float3 viewDir, float3 lightDir, float roughness) {
    float nDotV = max(dot(normal, viewDir), 0.0);
    float nDotL = max(dot(normal, lightDir), 0.0);
    return geometrySchlickGGX(nDotV, roughness) * geometrySchlickGGX(nDotL, roughness);
}

kernel void integrate_brdf(texture2d<float, access::write> brdfLut [[texture(0)]],
                           uint2 threadPos [[thread_position_in_grid]]) {
    uint width = brdfLut.get_width(), height = brdfLut.get_height();
    if (threadPos.x >= width || threadPos.y >= height) return;

    float2 texCoord = (float2(threadPos) + 0.5) / float2(width, height);
    float nDotV     = texCoord.x;
    float roughness = texCoord.y;

    float3 viewDir = float3(sqrt(1.0 - nDotV), 0.0, nDotV);
    float3 normal  = float3(0.0, 0.0, 1.0);

    float scale = 0.0;
    float bias  = 0.0;
    const uint SAMPLE_COUNT = 1024u;
    for (uint sampleIndex = 0u; sampleIndex < SAMPLE_COUNT; sampleIndex++) {
        float2 randomSample = hammersley(sampleIndex, SAMPLE_COUNT);
        float3 halfway      = importanceSampleGGX(randomSample, normal, roughness);
        float3 lightDir     = normalize(2.0 * dot(viewDir, halfway) * halfway - viewDir);

        float nDotL = max(lightDir.z, 0.0);
        float nDotH = max(halfway.z, 0.0);
        float vDotH = max(dot(viewDir, halfway), 0.0);
        if (nDotL > 0.0) {
            float geometry     = geometrySmith(normal, viewDir, lightDir, roughness);
            float geometryVis   = (geometry * vDotH) / (nDotH * nDotV);
            float fresnelFactor = pow(1.0 - vDotH, 5.0);
            scale += (1.0 - fresnelFactor) * geometryVis;
            bias  += fresnelFactor * geometryVis;
        }
    }
    brdfLut.write(float4(scale / SAMPLE_COUNT, bias / SAMPLE_COUNT, 0.0, 0.0), threadPos);
}

kernel void convolve_radiance(texturecube<float, access::sample>  environment [[texture(0)]],
                              texturecube<float, access::write>   radianceMap [[texture(1)]],
                              constant float& roughness           [[buffer(0)]],
                              uint3 threadPos [[thread_position_in_grid]]) {
    uint width = radianceMap.get_width(), height = radianceMap.get_height();
    if (threadPos.x >= width || threadPos.y >= height || threadPos.z >= 6) return;

    float2 uv     = float2(threadPos.xy) / float2(width, height);
    float3 normal = cubemapDirectionFromCoords(threadPos.z, uv);
    // for the split-sum prefilter, view and reflection are assumed equal to N
    float3 viewDir = normal;

    const uint SAMPLE_COUNT = 1024u;
    float  totalWeight  = 0.0;
    float3 prefiltered  = float3(0.0);
    for (uint sampleIndex = 0u; sampleIndex < SAMPLE_COUNT; sampleIndex++) {
        float2 randomSample = hammersley(sampleIndex, SAMPLE_COUNT);
        float3 halfway      = importanceSampleGGX(randomSample, normal, roughness);
        float3 lightDir     = normalize(2.0 * dot(viewDir, halfway) * halfway - viewDir);
        float  NdotL        = max(dot(normal, lightDir), 0.0);
        if (NdotL > 0.0) {
            prefiltered += environment.sample(linearSampler, lightDir).rgb * NdotL;
            totalWeight += NdotL;
        }
    }
    radianceMap.write(float4(prefiltered / max(totalWeight, 1e-4), 1.0), threadPos.xy, threadPos.z);
}

struct SkyboxOut {
    float4 position [[position]];
    float2 uv;
};

vertex SkyboxOut skybox_vs(uint vertexID [[vertex_id]]) {
    float2 uv = float2(float((vertexID << 1) & 2), float(vertexID & 2));
    SkyboxOut output;
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv       = uv;
    return output;
}


fragment float4 skybox_fs(SkyboxOut in [[stage_in]],
                          texturecube<float, access::sample> environment [[texture(0)]],
                          constant float4x4& invViewProj [[buffer(0)]]) {

    float2 ndc = in.uv * 2.0 - 1.0;

    float4 nearPoint = invViewProj * float4(ndc, 1.0, 1.0);
    float4 farPoint  = invViewProj * float4(ndc, 0.0, 1.0);
    float3 direction = normalize(farPoint.xyz / farPoint.w - nearPoint.xyz / nearPoint.w);

    // alpha 0 = far/background, so the temporal resolve treats sky as static under the fixed camera
    return float4(environment.sample(linearSampler, direction).rgb, 0.0);
}
