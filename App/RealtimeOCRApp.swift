import SwiftUI

@main
struct RealtimeOCRApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraPreview(camera: camera, isFrozen: camera.isFrozen).ignoresSafeArea()
            panel
        }
        .onAppear { camera.start() }
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let message = camera.errorMessage {
                Text(message).foregroundStyle(.red)
            }
            Text(camera.stableText.isEmpty ? "Fit one printed line inside the yellow box" : camera.stableText)
                .font(.title2)
                .foregroundStyle(camera.stableText.isEmpty ? .secondary : .primary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
            diagnostics
            Picker("OCR rate", selection: $camera.rate) {
                ForEach(CameraController.Rate.allCases) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)
            HStack {
                Button(camera.isFrozen ? "Resume" : "Freeze") { camera.isFrozen.toggle() }
                Button("Copy") { UIPasteboard.general.string = camera.stableText }
                    .disabled(camera.stableText.isEmpty)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding()
    }

    /// What the model saw last, what it read, and where the time went.
    @ViewBuilder private var diagnostics: some View {
        let status = camera.status
        if let crop = status.crop {
            Image(uiImage: crop).resizable().scaledToFit().frame(maxHeight: 36)
        }
        if let reading = status.reading {
            Text(String(format: "last %.2f%@: %@", reading.confidence, reading.isAcceptable ? "" : " (filtered)", reading.text))
                .font(.caption)
                .lineLimit(1)
            Text(String(
                format: "%.0f×%.0f px · crop %.1f · prep %.1f · infer %.1f · decode %.1f · total %.1f ms",
                status.cropSize.width, status.cropSize.height, status.cropMs,
                reading.prepMs, reading.inferMs, reading.decodeMs, status.totalMs
            ))
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }
}
