import os
import random
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


def collect_audio_pairs(dataset_path, genres=None):
    """
    Collects audio pairs from all genres in the GTZAN dataset

    Parameters:
        dataset_path: path to the main GTZAN folder
        genres: list of genres to train (None = all available)

    Returns:
        audio_pairs: list of tuples (raw_path, ref_path, filename, genre)
    """
    audio_pairs = []

    if genres is None:
        genres = [d for d in os.listdir(dataset_path) 
                 if os.path.isdir(os.path.join(dataset_path, d))]
    
    print(f"Searching for audio files in genres: {genres}")
    
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
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


def train_eq_agent(
    agent, 
    action_space, 
    dataset_path, 
    genres=None,
    num_episodes=1000, 
    max_steps_per_episode=20, 
    early_stopping_patience=5,
    save_path='./result/train_results',
    log_interval=10,
    feature_types=None,
    create_dataset=True,
    genre_balanced=True
):
    """
    Training an agent on multi-genre audio dataset
    
    Parameters:
        agent: an instance of the EQAgent class
        action_space: an instance of the EQActionSpace class
        dataset_path: path to the GTZAN dataset
        genres: list of genres to include (None = all available)
        num_episodes: maximum number of episodes
        max_steps_per_episode: maximum number of steps per episode
        early_stopping_patience: number of episodes without improvement
        save_path: path to save the model
        log_interval: interval for logging
        feature_types: types of features to extract
        create_dataset: flag to create dataset
        genre_balanced: if True, balance episodes across genres
    
    Returns:
        training_stats: dataframe with training statistics
    """
    
    if feature_types is None:
        feature_types = ['mel_spectrogram', 'mfcc', 'chroma']

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/spectrograms", exist_ok=True)
    os.makedirs(f"{save_path}/eq_responses", exist_ok=True)
    os.makedirs(f"{save_path}/processed_audio", exist_ok=True)

    audio_pairs = collect_audio_pairs(dataset_path, genres)
    
    print(f"Total found {len(audio_pairs)} pairs of audio files for training")

    if len(audio_pairs) == 0:
        raise ValueError("No audio file pairs found for training. Please check your data directory structure.")

    genre_groups = {}
    for pair in audio_pairs:
        genre = pair[3]
        if genre not in genre_groups:
            genre_groups[genre] = []
        genre_groups[genre].append(pair)
    
    print("Distribution by genre:")
    for genre, pairs in genre_groups.items():
        print(f"  {genre}: {len(pairs)} pairs")

    stats = {
        'episode': [],
        'pair_index': [],
        'genre': [],
        'final_reward': [],
        'spectral_distance_improvement': [],
        'perceptual_improvement': [],
        'energy_balance_improvement': [],
        'spectral_centroid_improvement': [],
        'spectral_flatness_improvement': [],
        'overall_quality': [],
        'steps': []
    }
    
    dataset = []
    for filter_type in ['low_shelf', 'peak_low', 'peak_high', 'high_shelf']:
        for param in ['freq', 'gain', 'q']:
            stats[f'{filter_type}_{param}'] = []
    
    best_quality = float('-inf')
    patience_counter = 0

    for episode in range(num_episodes):
        if genre_balanced and len(genre_groups) > 1:
            genre = random.choice(list(genre_groups.keys()))
            pair = random.choice(genre_groups[genre])
            pair_index = audio_pairs.index(pair)
        else:
            pair_index = random.randint(0, len(audio_pairs) - 1)
            pair = audio_pairs[pair_index]
        
        raw_path, ref_path, filename, genre = pair

        try:
            raw_audio, sr = librosa.load(raw_path, sr=None)
            ref_audio, _ = librosa.load(ref_path, sr=None)
        except Exception as e:
            print(f"Error loading audio files: {e}")
            continue

        raw_audio = raw_audio / np.max(np.abs(raw_audio))
        ref_audio = ref_audio / np.max(np.abs(ref_audio))

        min_length = min(len(raw_audio), len(ref_audio))
        raw_audio = raw_audio[:min_length]
        ref_audio = ref_audio[:min_length]
        
        state = extract_feature_vector(raw_audio, ref_audio, sr, feature_types)
        features_raw_audio = extract_features_for_eq_prediction(raw_audio, sr)
        
        total_reward = 0
        best_step_reward = float('-inf')
        best_step_metrics = None
        best_step_params = None
        best_processed_audio = None
        
        for step in range(max_steps_per_episode):
            action_indices = agent.select_action(state)
            params = actions_to_parameters(action_indices, action_space)
            processed_audio = apply_equalizer(raw_audio, sr, params)
            metrics = evaluate_quality(raw_audio, processed_audio, ref_audio, sr)
            reward = metrics['reward']
            
            if reward > best_step_reward:
                best_step_reward = reward
                best_step_metrics = metrics
                best_step_params = params
                best_processed_audio = processed_audio
            
            next_state = extract_feature_vector(processed_audio, ref_audio, sr, feature_types)
            done = (step == max_steps_per_episode - 1) or (reward > 0.95)
            
            agent.store_transition(state, action_indices, reward, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            if done:
                break

        stats['episode'].append(episode + 1)
        stats['pair_index'].append(pair_index)
        stats['genre'].append(genre)
        stats['final_reward'].append(best_step_reward)
        stats['spectral_distance_improvement'].append(best_step_metrics['spectral_distance']['improvement'])
        stats['perceptual_improvement'].append(best_step_metrics['perceptual_similarity']['improvement'])
        stats['energy_balance_improvement'].append(best_step_metrics['energy_balance']['improvement'])
        stats['spectral_centroid_improvement'].append(best_step_metrics['spectral_centroid']['improvement'])
        stats['spectral_flatness_improvement'].append(best_step_metrics['spectral_flatness']['improvement'])
        stats['overall_quality'].append(best_step_metrics['overall_quality'])
        stats['steps'].append(step + 1)

        record = {
            'file_path': raw_path,
            'file_name': filename,
            'genre': genre,
            'reward': best_step_reward,
            'overall_quality': best_step_metrics['overall_quality']
        }
        
        for feature_name, feature_value in features_raw_audio.items():
            record[f'feature_{feature_name}'] = feature_value
        
        for filter_type in ['low_shelf', 'peak_low', 'peak_high', 'high_shelf']:
            for param in ['freq', 'gain', 'q']:
                stats[f'{filter_type}_{param}'].append(best_step_params[f'{filter_type}_{param}'])
                record[f'eq_{filter_type}_{param}'] = best_step_params[f'{filter_type}_{param}']
        
        dataset.append(record)
        agent.reward_history.append(best_step_reward)

        if (episode + 1) % log_interval == 0 or episode == 0:
            print(f"Episode {episode + 1}/{num_episodes}, Genre: {genre}, "
                  f"Reward: {best_step_reward:.4f}, "
                  f"Quality: {best_step_metrics['overall_quality']:.4f}, "
                  f"Steps: {step + 1}")
            
            fig = plot_spectrograms(raw_audio, best_processed_audio, ref_audio, sr,
                                  title=f"Episode {episode + 1} - {genre}")
            plt.savefig(f"{save_path}/spectrograms/episode_{episode + 1}_{genre}.png")
            plt.close(fig)
            
            fig = plot_eq_response(sr, best_step_params)
            plt.savefig(f"{save_path}/eq_responses/episode_{episode + 1}_{genre}.png")
            plt.close(fig)
            
            sf.write(f"{save_path}/processed_audio/processed_episode_{episode + 1}_{genre}.wav", 
                    best_processed_audio, sr)
        
        # Early stopping
        current_quality = best_step_metrics['overall_quality']
        if current_quality > best_quality:
            best_quality = current_quality
            patience_counter = 0
            agent.save_model(f"{save_path}/best_model")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stop on episode {episode + 1}")
                break

    agent.save_model(f"{save_path}/final_model")
    training_stats = pd.DataFrame(stats)
    training_stats.to_csv(f"{save_path}/training_stats.csv", index=False)

    genre_stats = training_stats.groupby('genre').agg({
        'final_reward': ['mean', 'std', 'count'],
        'overall_quality': ['mean', 'std']
    }).round(4)

    genre_stats.columns = ['_'.join(col).strip() for col in genre_stats.columns.values]
    genre_stats.to_csv(f"{save_path}/genre_stats.csv")

    print("\nGenre Statistics:")
    print(genre_stats)
    
    fig = agent.plot_progress()
    plt.savefig(f"{save_path}/learning_progress.png")
    plt.close(fig)
    
    if create_dataset:
        dataset_df = pd.DataFrame(dataset)
        csv_path = f"{save_path}/eq_prediction_dataset.csv"
        dataset_df.to_csv(csv_path, index=False)
        
        print(f"\nDataset saved to {csv_path}")
        print(f"Dataset size: {len(dataset_df)} records, {len(dataset_df.columns)} features")
        print(f"Genres in dataset: {sorted(dataset_df['genre'].unique())}")
    
    return training_stats
