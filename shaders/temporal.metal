#include <metal_stdlib>
#include "pbr_common.h"
using namespace metal;

struct TemporalUniforms {
    float4x4 viewProj;
    float4x4 invViewProj;
    float4x4 deltaRotation;
    uint     historyValid;
};

constexpr sampler historySampler(coord::normalized, filter::linear, address::clamp_to_edge);

static inline float3 reconstructWorldPosition(float2 ndc, float depth, float4x4 invViewProj) {
    float4 worldPosition = invViewProj * float4(ndc, depth, 1.0);
    return worldPosition.xyz / worldPosition.w;
}

kernel void temporal_resolve(texture2d<float, access::read>    currentColor [[texture(0)]],
                             texture2d<float, access::sample>  historyColor [[texture(1)]],
                             texture2d<float, access::write>   outputColor  [[texture(2)]],
                             texture2d<float, access::write>   outputHistory[[texture(3)]],
                             constant TemporalUniforms&        frameData    [[buffer(0)]],
                             uint2                             gid          [[thread_position_in_grid]]) {
    uint2 texSize = uint2(currentColor.get_width(), currentColor.get_height());
    if (gid.x >= texSize.x || gid.y >= texSize.y) return;

    float4 currentSample    = currentColor.read(gid);
    float3 sampleColor      = currentSample.rgb;
    float  depth            = currentSample.a;

    float2 currentUv  = (float2(gid) + 0.5) / float2(texSize);
    float2 currentNdc = float2(currentUv.x * 2.0 - 1.0, 1.0 - currentUv.y * 2.0);
    bool   hasGeometry = depth > 0.0;

    float3 resultColor = sampleColor;

    if (frameData.historyValid != 0) {
        float3 currentWorldPosition = 0.0;
        float2 prevUv, prevNdc;
        if (hasGeometry) {
            currentWorldPosition     = reconstructWorldPosition(currentNdc, depth, frameData.invViewProj);
            float3 prevWorldPosition = (frameData.deltaRotation * float4(currentWorldPosition, 1.0)).xyz;
            float4 prevClip          = frameData.viewProj * float4(prevWorldPosition, 1.0);
            prevNdc = prevClip.xy / prevClip.w;
            prevUv  = prevNdc * float2(0.5, -0.5) + 0.5;
        } else {
            // background is static for the fixed camera
            prevUv  = currentUv;
            prevNdc = currentNdc;
        }

        bool validHistory = all(prevUv >= 0.0) && all(prevUv <= 1.0);
        if (validHistory) {
            float4 historySample = historyColor.sample(historySampler, prevUv);
            float  temporalWeight = 0.0f;
            if (hasGeometry) {
                float3 prevWorldPosition    = (frameData.deltaRotation * float4(currentWorldPosition, 1.0)).xyz;
                float3 historyWorldPosition = reconstructWorldPosition(prevNdc, historySample.a, frameData.invViewProj);
                float  worldSpaceDiff = distance(prevWorldPosition, historyWorldPosition);
                float  tolerance = 0.05;
                temporalWeight = 1.0 - saturate(worldSpaceDiff / tolerance);
                temporalWeight = temporalWeight * temporalWeight;
            }

            // bound the history sample to the current frame's local 3x3 colour box
            float3 boxMin = sampleColor;
            float3 boxMax = sampleColor;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int2   coord    = clamp(int2(gid) + int2(dx, dy), int2(0), int2(texSize) - 1);
                    float3 neighbor = currentColor.read(uint2(coord)).rgb;
                    boxMin = min(boxMin, neighbor);
                    boxMax = max(boxMax, neighbor);
                }
            }
            float3 clampedHistory = clamp(historySample.rgb, boxMin, boxMax);

            float blendFactor = temporalWeight * 0.95;
            resultColor = mix(sampleColor, clampedHistory, blendFactor);
        }
    }

    outputHistory.write(float4(resultColor, depth), gid);
    outputColor.write(float4(tonemapToSrgb(resultColor), 1.0), gid);
}
