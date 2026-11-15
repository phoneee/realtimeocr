import SwiftUI

@main
struct RealtimeOCRApp: App {
    var body: some Scene {
        WindowGroup { ContentView() }
    }
}

struct ContentView: View {
    @StateObject private var camera = CameraController()

    private var text: String { camera.status.reading?.text ?? "" }

    var body: some View {
        ZStack(alignment: .bottom) {
            CameraPreview(
                camera: camera, isFrozen: camera.isFrozen,
                boxes: camera.status.reading?.lines.map(\.quad) ?? [], frameSize: camera.status.frameSize
            )
            .ignoresSafeArea()
            panel
        }
        .onAppear { camera.start() }
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let message = camera.errorMessage {
                Text(message).foregroundStyle(.red)
            }
            ScrollView {
                Text(text.isEmpty ? "Point the camera at printed text" : text)
                    .font(.body)
                    .foregroundStyle(text.isEmpty ? .secondary : .primary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(maxHeight: 260)
            diagnostics
            Picker("Detail", selection: $camera.detail) {
                ForEach(CameraController.Detail.allCases) { Text("\($0.label) \($0.rawValue)").tag($0) }
            }
            .pickerStyle(.segmented)
            HStack {
                Button(camera.isFrozen ? "Resume" : "Freeze") { camera.isFrozen.toggle() }
                Button("Copy") { UIPasteboard.general.string = text }
                    .disabled(text.isEmpty)
            }
            .buttonStyle(.borderedProminent)
        }
        .padding()
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
        .padding()
    }

    /// How many lines were found and kept, and where the time went.
    @ViewBuilder private var diagnostics: some View {
        let status = camera.status
        if let reading = status.reading {
            Text(String(
                format: "%d lines kept of %d found · %.0f×%.0f px · frame %.0f · detect %.0f · read %.0f · total %.0f ms",
                reading.lines.count, reading.detected, status.frameSize.width, status.frameSize.height,
                status.renderMs, reading.detMs, reading.recMs, status.totalMs
            ))
            .font(.caption2.monospacedDigit())
            .foregroundStyle(.secondary)
        }
    }
}
