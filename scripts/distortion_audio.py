import os
import sys
import csv
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.audio_equalizer import shelf_filter, biquad_peaking_filter

def process_audio_files(input_dir, output_dir):

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    output_path.mkdir(parents=True, exist_ok=True)
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    all_track_params = {}
    
    if not input_path.exists():
        print(f"Directory {input_path} does not exist, skipping...")
        return {}

    files = [f for f in input_path.iterdir() 
             if f.is_file() and f.suffix.lower() in supported_formats]
    
    if not files:
        print(f"No audio files in {input_path}")
        return {}
    
    for file_path in files:
        filename = file_path.name
        
        try:
            audio, sr = librosa.load(str(file_path), sr=None)
            distorted_audio, eq_params = distort_audio(audio, sr)
            output_filename = file_path.stem + "_distorted.wav"
            output_file_path = output_path / output_filename
            sf.write(str(output_file_path), distorted_audio, sr)
            all_track_params[filename] = eq_params

            print("Equalizer settings:")
            for i, param in enumerate(eq_params):
                ftype = param["type"]
                freq = param["freq"]
                gain = param["gain"]
                q = param["q"]
                print(f"   {i+1}. {ftype}: freq={freq}Hz, gain={gain}dB, Q={q}")
            
        except Exception as e:
            print(f"Error processing file{filename}: {str(e)}")
    
    if all_track_params:
        csv_path = output_path / "distortion_parameters.csv"
        save_parameters_to_csv(all_track_params, csv_path)
        print(f"Settings saved in: {csv_path}")
    
    return all_track_params


def save_parameters_to_csv(all_track_params, csv_path):

    max_params = max([len(params) for params in all_track_params.values()]) if all_track_params else 0
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['filename']
        
        for i in range(max_params):
            fieldnames.extend([
                f'filter{i+1}_type', 
                f'filter{i+1}_freq', 
                f'filter{i+1}_gain', 
                f'filter{i+1}_q'
            ])
        
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for filename, params in all_track_params.items():
            row = {'filename': filename}
            for i, param in enumerate(params):
                row[f'filter{i+1}_type'] = param['type']
                row[f'filter{i+1}_freq'] = param['freq']
                row[f'filter{i+1}_gain'] = param['gain']
                row[f'filter{i+1}_q'] = param['q']
            
            writer.writerow(row)


def process_all_genres(gtzan_path):

    gtzan_path = Path(gtzan_path)
    
    if not gtzan_path.exists():
        print(f"Folder not found: {gtzan_path}")
        return

    genre_folders = [d for d in gtzan_path.iterdir() if d.is_dir()]
    
    if not genre_folders:
        print(f"No genre folders found in {gtzan_path}")
        return
    
    genre_names = [g.name for g in genre_folders]
    print(f"Found {len(genre_folders)} genres: {', '.join(genre_names)}")
    
    total_processed = 0
    successful_genres = []
    
    for genre_path in genre_folders:
        genre = genre_path.name
        
        reference_path = genre_path / "reference"
        raw_path = genre_path / "raw"

        result = process_audio_files(reference_path, raw_path)
        
        if result:
            files_count = len(result)
            total_processed += files_count
            successful_genres.append(f"{genre} ({files_count} files)")
            print(f"Genre {genre} successfully processed: {files_count} files")
        else:
            print(f"Genre {genre} skipped (no files or error)")


def distort_audio(audio, sr):

    params = [
        ("low", np.random.uniform(20, 600), np.random.uniform(-10, 10), np.random.uniform(0.7, 1.5)),
        ("peak", np.random.uniform(600, 6000), np.random.uniform(-10, 10), np.random.uniform(0.7, 1.5)),
        ("peak", np.random.uniform(6000, 12000), np.random.uniform(-10, 10), np.random.uniform(0.7, 1.5)),
        ("high", np.random.uniform(12000, 20000), np.random.uniform(-10, 10), np.random.uniform(0.7, 1.5)),
    ]
    
    distorted_audio = audio.copy()
    final_params = []
    
    for ftype, freq, gain, q_or_slope in params:
        before_max = np.max(np.abs(distorted_audio))
        
        if ftype == "low":
            filtered = shelf_filter(distorted_audio, sr, freq, gain, q_or_slope, "low")
        elif ftype == "high":
            filtered = shelf_filter(distorted_audio, sr, freq, gain, q_or_slope, "high")
        else:
            filtered = biquad_peaking_filter(distorted_audio, sr, freq, gain, q_or_slope)
            
        if np.array_equal(filtered, distorted_audio):
            print(f"Filter {ftype} skipped")
            continue
            
        distorted_audio = filtered
        after_max = np.max(np.abs(distorted_audio))
        if after_max > 1.5 * before_max and after_max > 0.1:
            distorted_audio = distorted_audio * (before_max / after_max)
        
        print(f"Filter {ftype}: freq={freq:.1f}Hz, gain={gain:.1f}dB, Q={q_or_slope:.2f}")
        
        final_params.append({
            "type": ftype,
            "freq": round(freq, 1),
            "gain": round(gain, 1),
            "q": round(q_or_slope, 2)
        })

    air_gain = np.random.uniform(1, 3)
    air_freq = np.random.uniform(12000, 16000)
    
    distorted_audio = biquad_peaking_filter(distorted_audio, sr, air_freq, air_gain, 0.7)

    max_val = np.max(np.abs(distorted_audio))
    if max_val > 0.01:
        distorted_audio = distorted_audio / max_val * 0.9
    
    return distorted_audio, final_params


def main():

    gtzan_directory = "rl-agent-audio-equalizer/data/GTZAN"
    process_all_genres(gtzan_directory)

if __name__ == "__main__":
    main()
