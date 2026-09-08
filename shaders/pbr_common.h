// MIT License
//
// Copyright (c) 2026 Kyriakos Gavras
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
