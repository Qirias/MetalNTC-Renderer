import Foundation
import AppKit
import Metal
import MetalKit


let benchmark = false

// Demo assets live beside this source file (assets/ in this submodule).
// #filePath keeps the paths correct wherever the repo is cloned, instead of a
// machine-specific absolute path.
let assetsDir    = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("assets")
let gltfURL      = assetsDir.appendingPathComponent("models/flighthelmet/scene.gltf")
let hdrURL       = assetsDir.appendingPathComponent("hdr/kloppenheim_06_4k.hdr")
let benchmarkNTC = assetsDir.appendingPathComponent("models/flighthelmet/GlassPlasticMat.ntc")

if !benchmark {
    guard FileManager.default.fileExists(atPath: gltfURL.path) else {
        fatalError("no .gltf at \(gltfURL.path)")
    }
}

guard let device = MTLCreateSystemDefaultDevice() else {
    fatalError("no Metal device")
}

let app = NSApplication.shared
app.setActivationPolicy(.regular)

let frame  = NSRect(x: 100, y: 100, width: 800, height: 800)
let window = NSWindow(contentRect: frame,
                     styleMask: [.titled, .closable, .resizable, .miniaturizable],
                     backing: .buffered,
                     defer:   false)
window.title = "NTC Renderer"

let mtkView = MTKView(frame: frame, device: device)
// bgra8Unorm (non-sRGB): the trainer sampled sRGB-encoded PNG values directly,
// so MLP outputs are already display-referred sRGB. Writing them into a
// non-sRGB drawable avoids double gamma encoding
mtkView.colorPixelFormat        = .bgra8Unorm
mtkView.depthStencilPixelFormat = .depth32Float
// the temporal-resolve compute pass writes the drawable directly, so it must not
// be framebuffer-only
mtkView.framebufferOnly         = false
mtkView.clearColor              = MTLClearColor(red: 0.08, green: 0.08, blue: 0.10, alpha: 1.0)
// reverse-z because matrix_perspective_right_hand is copied from my engine
mtkView.clearDepth              = 0.0
mtkView.preferredFramesPerSecond = 60

let renderer = try Renderer(view: mtkView, device: device, gltfURL: gltfURL, hdrURL: hdrURL,
                            benchmark: benchmark, benchmarkNTC: benchmarkNTC)
mtkView.delegate = renderer

window.contentView = mtkView
window.makeKeyAndOrderFront(nil)
window.center()

app.activate(ignoringOtherApps: true)
app.run()
