# MetalNTC Renderer

The optional real-time viewer + demo assets for [MetalNTC](https://github.com/Qirias/MetalNTC).

This repository is consumed as a git submodule at `sources/NTCRenderer` of the
main MetalNTC package. On its own it does not build — it depends on the
`NTCCore`, `NTCShared`, and `AAPLMath` targets from the parent package.

Contents:
- `*.swift`, `shaders/` — the `NTCRenderer` executable target (draws a glTF mesh
  whose fragment shader runs the NTC decoder per pixel; PBR/IBL + temporal STF
  resolve).
- `assets/models/` — demo meshes (FlightHelmet, SciFiHelmet) with their trained
  `.ntc` files beside each glTF.
- `assets/hdr/` — the environment map used for image-based lighting.

## Use

From a fresh clone of the parent repo:

```sh
git clone --recurse-submodules https://github.com/Qirias/MetalNTC.git
```

or, in an existing clone:

```sh
git submodule update --init sources/NTCRenderer
```

Then build the `NTCRenderer` product (Xcode, so the Metal shaders compile into
`default.metallib`). Without the submodule, the parent package simply omits the
`NTCRenderer` target and builds the trainer alone.
