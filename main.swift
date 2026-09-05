import Foundation
import AppKit
import Metal
import MetalKit

let benchmark = false
let benchmarkTensorOps = false

let assetsDir    = URL(fileURLWithPath: #filePath).deletingLastPathComponent().appendingPathComponent("assets")
let gltfURL      = assetsDir.appendingPathComponent("models/flighthelmet/scene.gltf")
let hdrURL       = assetsDir.appendingPathComponent("hdr/kloppenheim_06_4k.hdr")
let blueNoiseURL = assetsDir.appendingPathComponent("hdr/LDR_RGB1_0.png")
let benchmarkNTC = assetsDir.appendingPathComponent("models/flighthelmet/GlassPlasticMat_high.ntc")

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

let frame  = NSRect(x: 100, y: 100, width: 1920, height: 1080)
let window = NSWindow(contentRect: frame,
                     styleMask: [.titled, .closable, .resizable, .miniaturizable],
                     backing: .buffered,
                     defer:   false)
window.title = "NTC Renderer"

/// MTKView that forwards orbit/zoom input to the renderer. Subclassing keeps
/// the events scoped to the view instead of a global event monitor that would
/// also fire while the picker has focus.
final class InputMTKView: MTKView {
    weak var renderer: Renderer?

    override var acceptsFirstResponder: Bool { true }

    override func mouseDragged(with event: NSEvent) {
        renderer?.rotate(deltaX: Float(event.deltaX), deltaY: Float(event.deltaY))
    }

    override func scrollWheel(with event: NSEvent) {
        // trackpads report precise deltas an order of magnitude larger than a
        // wheel notch, so normalise before handing it over
        let delta = event.hasPreciseScrollingDeltas ? event.scrollingDeltaY * 0.1
                                                    : event.scrollingDeltaY
        renderer?.zoom(delta: Float(delta))
    }
}

let mtkView = InputMTKView(frame: frame, device: device)
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
                            blueNoiseURL: blueNoiseURL,
                            benchmark: benchmark, benchmarkTensorOps: benchmarkTensorOps,
                            benchmarkNTC: benchmarkNTC)
mtkView.delegate = renderer
mtkView.renderer = renderer

// controls overlay the top-left of the view, so the MTKView goes in a
// container rather than being the contentView itself
let container = NSView(frame: frame)
// the MTKView is layer-backed; without this the sibling controls can end up
// behind it regardless of subview order
container.wantsLayer = true
mtkView.frame = container.bounds
mtkView.autoresizingMask = [.width, .height]
container.addSubview(mtkView)

@MainActor
func checkbox(_ title: String, _ on: Bool, _ action: Selector) -> NSButton {
    let b = NSButton(checkboxWithTitle: title, target: renderer, action: action)
    b.state = on ? .on : .off
    b.contentTintColor = .white
    return b
}

var controls: [NSView] = []
// one segment per quality that has a .ntc for every material
let qualities = renderer.availableQualities
if qualities.count > 1 {
    let picker = NSSegmentedControl(labels: qualities.map(\.rawValue),
                                    trackingMode: .selectOne,
                                    target: renderer,
                                    action: #selector(Renderer.qualityChanged(_:)))
    picker.selectedSegment = renderer.activeQualityIndex
    controls.append(picker)
}
controls.append(checkbox("Stochastic LOD", renderer.stochasticLOD,
                         #selector(Renderer.stochasticLODChanged(_:))))
controls.append(checkbox("Temporal", renderer.temporalAccumulation,
                         #selector(Renderer.temporalChanged(_:))))
controls.append(checkbox("Auto-rotate", renderer.autoRotate,
                         #selector(Renderer.autoRotateChanged(_:))))

let panel = NSStackView(views: controls)
panel.orientation = .vertical
panel.alignment   = .leading
panel.spacing     = 6
panel.edgeInsets  = NSEdgeInsets(top: 8, left: 10, bottom: 8, right: 10)
panel.wantsLayer  = true
panel.layer?.backgroundColor = NSColor.black.withAlphaComponent(0.45).cgColor
panel.layer?.cornerRadius    = 8
panel.setFrameSize(panel.fittingSize)
// pinned top-left; .minYMargin holds it there as the window grows
panel.setFrameOrigin(NSPoint(x: 12, y: container.bounds.height - panel.frame.height - 12))
panel.autoresizingMask = [.minYMargin]
container.addSubview(panel)

window.contentView = container
window.makeFirstResponder(mtkView)
window.makeKeyAndOrderFront(nil)
window.center()

app.activate(ignoringOtherApps: true)
app.run()
