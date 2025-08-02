import librosa
import numpy as np
from src.core.audio_metrics import perceptual_eval, spectral_distance
from src.utils.feature_extraction import compute_band_energy


def evaluate_quality(raw_audio, processed_audio, reference_audio, sr):
    """
    Advanced audio processing quality assessment using multiple metrics

    Parameters:
        raw_audio: raw audio
        processed_audio: processed audio
        reference_audio: reference audio
        sr: sampling rate

    Returns:
        metrics: dictionary with various quality assessment metrics
    """

    raw_ref_distance = spectral_distance(raw_audio, reference_audio, sr)
    proc_ref_distance = spectral_distance(processed_audio, reference_audio, sr)

    raw_ref_similarity = perceptual_eval(raw_audio, reference_audio, sr)
    proc_ref_similarity = perceptual_eval(processed_audio, reference_audio, sr)

    spectral_improvement = (raw_ref_distance - proc_ref_distance) / (raw_ref_distance + 1e-8)
    perceptual_improvement = (proc_ref_similarity - raw_ref_similarity) / (1 - raw_ref_similarity + 1e-8)

    bands = [(20, 600), (600, 6000), (6000, 12000), (12000, 20000)]
    band_names = ['bass', 'mid', 'upper_mid', 'treble']
    
    raw_energy = {}
    proc_energy = {}
    ref_energy = {}
    
    for band_name, (low, high) in zip(band_names, bands):
        raw_energy[band_name] = compute_band_energy(raw_audio, sr, low, high)
        proc_energy[band_name] = compute_band_energy(processed_audio, sr, low, high)
        ref_energy[band_name] = compute_band_energy(reference_audio, sr, low, high)

    energy_balance_raw = np.array([raw_energy[band] / (sum(raw_energy.values()) + 1e-8) for band in band_names])
    energy_balance_proc = np.array([proc_energy[band] / (sum(proc_energy.values()) + 1e-8) for band in band_names])
    energy_balance_ref = np.array([ref_energy[band] / (sum(ref_energy.values()) + 1e-8) for band in band_names])
    
    raw_balance_error = np.sum((energy_balance_raw - energy_balance_ref)**2)
    proc_balance_error = np.sum((energy_balance_proc - energy_balance_ref)**2)
    balance_improvement = (raw_balance_error - proc_balance_error) / (raw_balance_error + 1e-8) if raw_balance_error > 0 else 0

    raw_centroid = np.mean(librosa.feature.spectral_centroid(y=raw_audio, sr=sr)[0])
    proc_centroid = np.mean(librosa.feature.spectral_centroid(y=processed_audio, sr=sr)[0])
    ref_centroid = np.mean(librosa.feature.spectral_centroid(y=reference_audio, sr=sr)[0])
    
    raw_centroid_error = abs(raw_centroid - ref_centroid) / ref_centroid
    proc_centroid_error = abs(proc_centroid - ref_centroid) / ref_centroid
    centroid_improvement = (raw_centroid_error - proc_centroid_error) / (raw_centroid_error + 1e-8) if raw_centroid_error > 0 else 0
    
    raw_flatness = np.mean(librosa.feature.spectral_flatness(y=raw_audio)[0])
    proc_flatness = np.mean(librosa.feature.spectral_flatness(y=processed_audio)[0])
    ref_flatness = np.mean(librosa.feature.spectral_flatness(y=reference_audio)[0])
    
    raw_flatness_error = abs(raw_flatness - ref_flatness) / (ref_flatness + 1e-8)
    proc_flatness_error = abs(proc_flatness - ref_flatness) / (ref_flatness + 1e-8)
    flatness_improvement = (raw_flatness_error - proc_flatness_error) / (raw_flatness_error + 1e-8) if raw_flatness_error > 0 else 0

    reward = (
        0.25 * spectral_improvement +  
        0.30 * perceptual_improvement + 
        0.20 * balance_improvement +    
        0.15 * centroid_improvement + 
        0.10 * flatness_improvement    
    )
    
    overall_quality = reward
    reward = min(max(reward, -1.0), 1.0)

    metrics = {
        'spectral_distance': {
            'raw_ref': raw_ref_distance,
            'proc_ref': proc_ref_distance,
            'improvement': spectral_improvement
        },
        'perceptual_similarity': {
            'raw_ref': raw_ref_similarity,
            'proc_ref': proc_ref_similarity,
            'improvement': perceptual_improvement
        },
        'energy_balance': {
            'raw_error': raw_balance_error,
            'proc_error': proc_balance_error,
            'improvement': balance_improvement
        },
        'spectral_centroid': {
            'raw_error': raw_centroid_error,
            'proc_error': proc_centroid_error,
            'improvement': centroid_improvement
        },
        'spectral_flatness': {
            'raw_error': raw_flatness_error,
            'proc_error': proc_flatness_error,
            'improvement': flatness_improvement
        },
        'overall_quality': overall_quality,
        'reward': reward
    }
    
    return metrics
