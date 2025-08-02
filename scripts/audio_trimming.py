import os
import librosa
import soundfile as sf
from pathlib import Path


def trim_audio_to_30_seconds(input_folder, output_folder):

    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Supported audio formats
    audio_extensions = {'.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    
    processed_count = 0
    error_count = 0
    
    print(f"Starting file processing from folder: {input_folder}")
    print(f"Results will be saved to: {output_folder}")
    print("-" * 60)

    for root, _, files in os.walk(input_folder):
        for file in files:
            file_path = Path(root) / file
            
            if file_path.suffix.lower() in audio_extensions:
                try:
                    print(f"Processing: {file}")

                    audio, sr = librosa.load(str(file_path), sr=None)

                    duration = len(audio) / sr
                    
                    if duration < 30:
                        print(f"  File {file} is shorter than 30 seconds ({duration:.1f}s), skipping")
                        continue

                    middle_sample = len(audio) // 2

                    samples_30_sec = int(30 * sr)
                    start_sample = middle_sample - samples_30_sec // 2
                    end_sample = start_sample + samples_30_sec
                    
                    if start_sample < 0:
                        start_sample = 0
                        end_sample = samples_30_sec
                    elif end_sample > len(audio):
                        end_sample = len(audio)
                        start_sample = end_sample - samples_30_sec

                    trimmed_audio = audio[start_sample:end_sample]

                    relative_path = Path(root).relative_to(input_folder)
                    output_subfolder = Path(output_folder) / relative_path
                    output_subfolder.mkdir(parents=True, exist_ok=True)

                    output_filename = f"{file_path.stem}_30sec{file_path.suffix}"
                    output_path = output_subfolder / output_filename

                    sf.write(str(output_path), trimmed_audio, sr)
                    
                    print(f"  Saved: {output_path}")
                    processed_count += 1
                    
                except Exception as e:
                    print(f"  Error processing {file}: {str(e)}")
                    error_count += 1
    
    print("-" * 60)
    print(f"Successfully processed: {processed_count} files")
    print(f"Errors: {error_count}")


def main():

    input_folder = "rl-agent-audio-equalizer/data/FAM"
    
    if not os.path.exists(input_folder):
        print("Folder does not exist!")
        return

    output_folder = os.path.join(os.path.dirname(input_folder), "trimmed_30sec")

    print(f"\Input folder: {input_folder}")
    print(f"Trim folder: {output_folder}")
    
    trim_audio_to_30_seconds(input_folder, output_folder)


if __name__ == "__main__":
    main()