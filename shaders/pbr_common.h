// MetalNTC Renderer — Copyright (c) 2026 Kyriakos Gavras
//
// Licensed under the PolyForm Noncommercial License 1.0.0.
// Noncommercial use only; released for research and education.
// See LICENSE.md, or https://polyformproject.org/licenses/noncommercial/1.0.0

#pragma once
#include <metal_stdlib>
using namespace metal;

inline float3 fresnelSchlick(float cosTheta, float3 baseReflectance) {
    return baseReflectance + (1.0 - baseReflectance) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

inline float3 fresnelSchlickRoughness(float cosTheta, float3 baseReflectance, float roughness) {
    return baseReflectance
         + (max(float3(1.0 - roughness), baseReflectance) - baseReflectance)
           * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

constant float PBR_EXPOSURE = 1.0;

inline float3 srgbToLinear(float3 color) { return pow(max(color, 0.0), 2.2); }        // cheap gamma approx
inline float3 linearToSrgb(float3 color) { return pow(max(color, 0.0), 1.0 / 2.2); }

// ACES filmic tone-map approximation (Krzysztof Narkowicz)
inline float3 acesFilm(float3 color) {
    return clamp((color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14), 0.0, 1.0);
}

inline float3 tonemapToSrgb(float3 hdrColor) {
    return linearToSrgb(acesFilm(hdrColor * PBR_EXPOSURE));
}
