// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "OCRCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [.library(name: "OCRCore", targets: ["OCRCore"])],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", exact: "1.24.2"),
    ],
    targets: [
        .target(
            name: "OCRCore",
            dependencies: [.product(name: "onnxruntime", package: "onnxruntime-swift-package-manager")],
            resources: [.copy("Model")]
        ),
        .testTarget(
            name: "OCRCoreTests",
            dependencies: ["OCRCore"],
            resources: [.copy("Fixtures")]
        ),
    ]
)
