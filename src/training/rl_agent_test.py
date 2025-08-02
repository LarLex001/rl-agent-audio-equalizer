import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
from src.utils.feature_extraction import extract_feature_vector, extract_features_for_eq_prediction
from src.agent.eq_action_space import actions_to_parameters 
from src.core.audio_equalizer import apply_equalizer 
from src.core.quality_evaluator import evaluate_quality 
from src.utils.plot_audio_eq import *


def collect_test_audio_pairs(test_path, genres=None):
    """
    Collects pairs of audio files for testing from all genres

    Parameters:
        test_path: path to the test dataset
        genres: list of genres to test (None = all available)

    Returns:
        audio_pairs: list of tuples (raw_path, ref_path, filename, genre)
    """

    audio_pairs = []

    if genres is None:
        genres = [d for d in os.listdir(test_path) 
                 if os.path.isdir(os.path.join(test_path, d))]
    
    print(f"Searching for test audio files in genres: {genres}")
    
    for genre in genres:
        genre_path = os.path.join(test_path, genre)
        raw_dir = os.path.join(genre_path, "raw")
        ref_dir = os.path.join(genre_path, "reference")
        
        if not os.path.exists(raw_dir) or not os.path.exists(ref_dir):
            print(f"Skipping {genre}: missing raw or reference directory")
            continue
        
        genre_pairs = 0
        for file in os.listdir(raw_dir):
            if file.endswith('.wav') or file.endswith('.mp3'):
                raw_path = os.path.join(raw_dir, file)
                ref_path = os.path.join(ref_dir, file)

                if not os.path.exists(ref_path):
                    ref_name = file.replace("_distorted", "")
                    ref_path = os.path.join(ref_dir, ref_name)
                
                if os.path.exists(ref_path):
                    audio_pairs.append((raw_path, ref_path, file, genre))
                    genre_pairs += 1
                else:
                    print(f"No matching reference file found for {genre}/{file}")
        
        print(f"Found {genre_pairs} pairs in {genre}")
    
    return audio_pairs


def test_eq_agent(
    agent, 
    action_space, 
    test_path, 
    genres=None,
    feature_types=None, 
    save_path='./test_results',
    save_processed_audio=True,
    save_visualizations=True,
    create_dataset=True,
):
    """
    Testing a trained agent on a multi-genre dataset

    Parameters:
        agent: trained instance of the EQAgent class
        action_space: instance of the EQActionSpace class
        test_path: path to the test dataset (GTZAN structure)
        genres: list of genres to test (None = all available)
        feature_types: types of features to extract from the audio
        save_path: path to save the results
        save_processed_audio: whether to save processed audio files
        save_visualizations: whether to save spectrograms and EQ responses

    Returns:
        test_stats: dataframe with test statistics
    """

    if feature_types is None:
        feature_types = ['mel_spectrogram', 'mfcc', 'chroma']

    os.makedirs(save_path, exist_ok=True)
    if save_visualizations:
        os.makedirs(f"{save_path}/spectrograms", exist_ok=True)
        os.makedirs(f"{save_path}/eq_responses", exist_ok=True)
    if save_processed_audio:
        os.makedirs(f"{save_path}/processed_audio", exist_ok=True)

    audio_pairs = collect_test_audio_pairs(test_path, genres)

    print(f"Total found {len(audio_pairs)} pairs of audio files for testing")
    
    if len(audio_pairs) == 0:
        raise ValueError("No audio file pairs found for testing. Please check your data directory structure.")

    genre_groups = {}
    for pair in audio_pairs:
        genre = pair[3]
        if genre not in genre_groups:
            genre_groups[genre] = []
        genre_groups[genre].append(pair)
    
    print("Test distribution by genre:")
    for genre, pairs in genre_groups.items():
        print(f"  {genre}: {len(pairs)} pairs")

    dataset = []
    stats = {
        'file_name': [],
        'genre': [],
        'reward': [],
        'spectral_improvement': [],
        'perceptual_improvement': [],
        'energy_balance_improvement': [],
        'spectral_centroid_improvement': [],
        'spectral_flatness_improvement': [],
        'overall_quality': []
    }

    for filter_type in ['low_shelf', 'peak_low', 'peak_high', 'high_shelf']:
        for param in ['freq', 'gain', 'q']:
            stats[f'{filter_type}_{param}'] = []

    for i, (raw_path, ref_path, file_name, genre) in enumerate(audio_pairs):
        print(f"Testing file {i+1}/{len(audio_pairs)}: {genre}/{file_name}")

        try:
            raw_audio, sr = librosa.load(raw_path, sr=None)
            ref_audio, _ = librosa.load(ref_path, sr=None)

            raw_audio = raw_audio / np.max(np.abs(raw_audio))
            ref_audio = ref_audio / np.max(np.abs(ref_audio))

            min_length = min(len(raw_audio), len(ref_audio))
            raw_audio = raw_audio[:min_length]
            ref_audio = ref_audio[:min_length]

            state = extract_feature_vector(raw_audio, ref_audio, sr, feature_types)
            features_raw_audio = extract_features_for_eq_prediction(raw_audio, sr)

            epsilon_backup = agent.epsilon
            agent.epsilon = 0  # disabling randomness for deterministic behavior
            action_indices = agent.select_action(state)
            agent.epsilon = epsilon_backup 

            params = actions_to_parameters(action_indices, action_space)
            processed_audio = apply_equalizer(raw_audio, sr, params)

            metrics = evaluate_quality(raw_audio, processed_audio, ref_audio, sr)

            stats['file_name'].append(file_name)
            stats['genre'].append(genre)
            stats['reward'].append(metrics['reward'])
            stats['spectral_improvement'].append(metrics['spectral_distance']['improvement'])
            stats['perceptual_improvement'].append(metrics['perceptual_similarity']['improvement'])
            stats['energy_balance_improvement'].append(metrics['energy_balance']['improvement'])
            stats['spectral_centroid_improvement'].append(metrics['spectral_centroid']['improvement'])
            stats['spectral_flatness_improvement'].append(metrics['spectral_flatness']['improvement'])
            stats['overall_quality'].append(metrics['overall_quality'])

            record = {
                'file_path': raw_path,
                'file_name': file_name,
                'genre': genre,
                'reward': metrics['reward'],
                'overall_quality': metrics['overall_quality']
            }
        
            for feature_name, feature_value in features_raw_audio.items():
                record[f'feature_{feature_name}'] = feature_value

            for filter_type in ['low_shelf', 'peak_low', 'peak_high', 'high_shelf']:
                for param in ['freq', 'gain', 'q']:
                    stats[f'{filter_type}_{param}'].append(params[f'{filter_type}_{param}'])
                    record[f'eq_{filter_type}_{param}'] = params[f'{filter_type}_{param}']

            dataset.append(record)
            base_name = os.path.splitext(file_name)[0]

            if save_visualizations:
                fig = plot_spectrograms(raw_audio, processed_audio, ref_audio, sr,
                                      title=f"Spectrograms for {genre}/{base_name}")
                plt.savefig(f"{save_path}/spectrograms/{genre}_{base_name}.png")
                plt.close(fig)

                fig = plot_eq_response(sr, params)
                plt.savefig(f"{save_path}/eq_responses/{genre}_{base_name}.png")
                plt.close(fig)

            if save_processed_audio:
                sf.write(f"{save_path}/processed_audio/{genre}_{base_name}_processed.wav", 
                        processed_audio, sr)

        except Exception as e:
            print(f"Error processing {genre}/{file_name}: {e}")
            continue

    test_stats = pd.DataFrame(stats)
    test_stats.to_csv(f"{save_path}/test_stats.csv", index=False)

    avg_quality = test_stats['overall_quality'].mean()
    avg_reward = test_stats['reward'].mean()
    
    print(f"\nOverall Results")
    print(f"Average quality: {avg_quality:.4f}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Total files tested: {len(test_stats)}")

    genre_stats = test_stats.groupby('genre').agg({
        'reward': ['mean', 'std', 'count'],
        'overall_quality': ['mean', 'std'],
        'spectral_improvement': 'mean',
        'perceptual_improvement': 'mean',
        'energy_balance_improvement': 'mean'
    }).round(4)

    genre_stats.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0] for col in genre_stats.columns]
    genre_stats = genre_stats.reset_index()
    genre_stats.to_csv(f"{save_path}/genre_test_stats.csv", index=False)
    
    print(f"\nGenre Breakdown")
    for genre in sorted(test_stats['genre'].unique()):
        genre_data = test_stats[test_stats['genre'] == genre]
        print(f"{genre}:")
        print(f"  Files: {len(genre_data)}")
        print(f"  Avg Quality: {genre_data['overall_quality'].mean():.4f}")
        print(f"  Avg Reward: {genre_data['reward'].mean():.4f}")
        print(f"  Spectral Improvement: {genre_data['spectral_improvement'].mean():.4f}")

    print(f"\nImprovement Metrics")
    print(f"Spectral Distance: {test_stats['spectral_improvement'].mean():.4f}")
    print(f"Perceptual Similarity: {test_stats['perceptual_improvement'].mean():.4f}")
    print(f"Energy Balance: {test_stats['energy_balance_improvement'].mean():.4f}")
    print(f"Spectral Centroid: {test_stats['spectral_centroid_improvement'].mean():.4f}")
    print(f"Spectral Flatness: {test_stats['spectral_flatness_improvement'].mean():.4f}")

    if create_dataset:
        dataset_df = pd.DataFrame(dataset)
        csv_path = f"{save_path}/eq_prediction_dataset.csv"
        dataset_df.to_csv(csv_path, index=False)
        
        print(f"\nDataset saved to {csv_path}")
        print(f"Dataset size: {len(dataset_df)} records, {len(dataset_df.columns)} features")
        print(f"Genres in dataset: {sorted(dataset_df['genre'].unique())}")
    
    return test_stats


def test_single_genre(agent, action_space, test_path, genre, **kwargs):
    """
    Testing on one genre (for backward compatibility)
    """

    return test_eq_agent(
        agent, action_space, test_path, 
        genres=[genre], **kwargs
    )
