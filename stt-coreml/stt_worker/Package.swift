// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "stt_worker",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        // pinned to the exact revision the demo binary was built from. tracking
        // "main" means any rebuild can silently change transcription behaviour,
        // which is not something to discover during a live session.
        .package(
            url: "https://github.com/FluidInference/FluidAudio.git",
            revision: "6428e29186573c6d33c598e25d460e6690bc0ee1"
        )
    ],
    targets: [
        .executableTarget(
            name: "stt_worker",
            dependencies: [
                .product(name: "FluidAudio", package: "FluidAudio")
            ]
        ),
    ]
)
