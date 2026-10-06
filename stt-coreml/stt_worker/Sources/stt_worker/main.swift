import Foundation
import FluidAudio

@main
struct STTWorker {
    static func main() async throws {
        // initialize the transcriber and load the neural engine model into ram
        // we explicitly use pktv3 (parakeet tdt v3)
        fputs("[[STT_WORKER]]: initializing FluidAudio CoreML model...\n", stderr)
        let tdtModels: AsrModels
        do {
            tdtModels = try await AsrModels.downloadAndLoad(version: .v3)
        } catch {
            fputs("[[STT_WORKER ERROR]]: Failed to download/load CoreML model: \(error)\n", stderr)
            throw error
        }
        let manager = AsrManager(config: .default)
        
        try await manager.loadModels(tdtModels)
        
        let converter = AudioConverter()

        fputs("[[STT_WORKER]]: ready\n", stderr)
        
        // loop continuously, waiting for python to pipe a file path
        while let filePath = readLine() {
            let path = filePath.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !path.isEmpty else { continue }
            
            let url = URL(fileURLWithPath: path)
            guard FileManager.default.fileExists(atPath: url.path) else {
                fputs("Error: File not found \(path)\n", stderr)
                continue
            }
            
            do {
                // 1. load and resample audio using fluidaudio's helper
                let samples = try converter.resampleAudioFile(url)
                
                // 2. setup the decoder state required for pktv3
                var decoderState = TdtDecoderState.make(decoderLayers: await manager.decoderLayerCount)
                
                // 3. transcribe instantly using the pre-loaded coreml model
                let result = try await manager.transcribe(samples, decoderState: &decoderState)
                
                // 4. print only the transcription to stdout for python to capture
                print(result.text)
                fflush(stdout)
                
            } catch {
                fputs("Transcription failed: \(error)\n", stderr)
                // print a blank line so python knows we finished processing but failed
                print("")
                fflush(stdout)
            }
        }
    }
}
