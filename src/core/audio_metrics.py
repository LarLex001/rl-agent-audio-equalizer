import librosa
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

## Basic distance metrics

def mel_spectrogram_mse(audio1, audio2, sr, n_mels=128, n_fft=2048, hop_length=512):
    """
    Calculates the mean square error between log-Mel spectrograms of two audio signals

    Parameters:
        audio1, audio2: audio signals to compare
        sr: sampling rate in Hz
        n_mels: number of Mel bands to generate
        n_fft: FFT size for spectrogram computation
        hop_length: window hopping step in samples

    Returns:
        mse: mean square error between log-Mel spectrograms
    """

    S1 = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio1, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length),
        ref=np.max
    )
    S2 = librosa.power_to_db(
        librosa.feature.melspectrogram(y=audio2, sr=sr, n_mels=n_mels,
                                       n_fft=n_fft, hop_length=hop_length),
        ref=np.max
    )
    return np.mean((S1 - S2) ** 2)


def spectral_convergence(S_ref, S_proc):
    """
    Calculates spectral convergence between two power spectrograms

    Parameters:
        S_ref: reference power spectrogram 
        S_proc: processed power spectrogram

    Returns:
        convergence: spectral convergence metric 
    """
    
    return np.linalg.norm(S_ref - S_proc, 'fro') / (np.linalg.norm(S_ref, 'fro') + 1e-8)


def multi_scale_spectral_loss(audio1, audio2, sr, scales=[512, 2048, 8192], hop_length=256):
    """
    Calculates multi-scale spectral convergence loss between two audio signals

    Parameters:
        audio1, audio2: audio signals to compare
        sr: sampling rate in Hz
        scales: list of FFT sizes for multi-scale analysis
        hop_length: window hopping step in samples

    Returns:
        loss: averaged spectral convergence across all scales 
    """

    losses = []
    for n_fft in scales:
        S_ref = librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length)
        S_proc = librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length)
        P_ref = np.abs(S_ref) ** 2
        P_proc = np.abs(S_proc) ** 2
        losses.append(spectral_convergence(P_ref, P_proc))
    return np.mean(losses)


def spectral_distance(audio1, audio2, sr, n_fft=2048, hop_length=512):
    """
    Calculates the spectral distance between two audio signals

    Parameters:
        audio1, audio2: audio signals to compare
        sr: sampling rate
        n_fft: FFT size
        hop_length: window hopping step

    Returns:
        mse: mean square error between spectrograms
    """

    S1 = np.abs(librosa.stft(audio1, n_fft=n_fft, hop_length=hop_length))
    S2 = np.abs(librosa.stft(audio2, n_fft=n_fft, hop_length=hop_length))

    S1_db = librosa.amplitude_to_db(S1, ref=np.max)
    S2_db = librosa.amplitude_to_db(S2, ref=np.max)

    mse = np.mean((S1_db - S2_db) ** 2)
    
    return mse


def perceptual_eval(audio1, audio2, sr):
    """
    Calculates a combined perceptual similarity between two audio signals:

    Parameters:
        audio1, audio2: audio signals to compare
        sr: sampling rate

    Returns:
        similarity score [0..1]
    """

    # DTW on MFCC
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr, n_mfcc=13)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr, n_mfcc=13)
    D, _ = fastdtw(mfcc1.T, mfcc2.T, dist=euclidean)
    dtw_dist = D / (mfcc1.shape[1] + mfcc2.shape[1])
    sim_a = 1 / (1 + dtw_dist)

    # Mel-spectrogram MSE
    mse_b = mel_spectrogram_mse(audio1, audio2, sr)
    sim_b = 1 / (1 + mse_b)

    # Multi-scale spectral convergence
    loss_c = multi_scale_spectral_loss(audio1, audio2, sr)
    sim_c = 1 / (1 + loss_c)

    weights = {'a': 0.4, 'b': 0.3, 'c': 0.3}
    total_sim = (
        weights['a'] * sim_a +
        weights['b'] * sim_b +
        weights['c'] * sim_c
    )

    return np.clip(total_sim, 0.0, 1.0)
