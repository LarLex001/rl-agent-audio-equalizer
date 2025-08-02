import librosa
import numpy as np

# Feature extraction

def compute_band_energy(audio, sr, low_freq, high_freq, n_fft=2048):
    """
    Calculates the energy in a given frequency band.

    Parameters:
      audio: audio signal
      sr: sampling frequency
      low_freq: lower limit of the frequency band (Hz)
      high_freq: upper limit of the frequency band (Hz)
      n_fft: FFT size

    Returns:
      energy: energy in a given frequency band
    """

    S = np.abs(librosa.stft(audio, n_fft=n_fft))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = np.logical_and(freqs >= low_freq, freqs <= high_freq)
    band_energy = np.sum(np.mean(S[mask], axis=1))
    
    return band_energy


def extract_features(audio, sr, feature_type='mel_spectrogram', n_mels=128, n_fft=2048, hop_length=512):
    """
    Extracts audio features from an audio signal

    Parameters:
      audio: audio signal
      sr: sampling rate
      feature_type: feature type ('mel_spectrogram', 'mfcc', 'chroma', 'spectral_contrast')
      n_mels: number of mel filters for mel_spectrogram
      n_fft: FFT size
      hop_length: hopping step

    Returns:
      features: matrix of extracted features
    """

    if feature_type == 'mel_spectrogram':
        # Mel-mel_spectrogram with logarithmic scale
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, 
                                          hop_length=hop_length, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)
        return S_dB
    
    elif feature_type == 'mfcc':
        # MFCC - Mel-Frequency Cepstral Coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
        mfccs_normalized = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / (np.std(mfccs, axis=1, keepdims=True) + 1e-8)
        return mfccs_normalized
    
    elif feature_type == 'chroma':
        # chromatogram represents the energy in each of the 12 semitones
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return chroma
    
    elif feature_type == 'spectral_contrast':
        # spectral contrast the difference between peaks and valleys in a spectrum
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
        return contrast
    
    elif feature_type == 'spectral_centroid':
        # spectral centroid represents the "center of mass" of the spectrum
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        return centroid
    
    elif feature_type == 'spectral_flatness':
        # spectral flatness measures how evenly energy is distributed across the spectrum
        flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
        return flatness 
 
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    

def extract_feature_vector(audio_raw, audio_ref, sr, feature_types=None):
    """
    Creates a feature vector for training an RL model by comparing the input and reference audio

    Parameters:
      audio_raw: raw audio
      audio_ref: reference audio
      sr: sampling rate
      feature_types: list of feature types to extract

    Returns:
      feature_vector: one-dimensional array of features
    """
    
    if feature_types is None:
       feature_types = ['mel_spectrogram', 'mfcc', 'chroma']
    
    features = []

    bands = [(20, 600), (600, 6000), (6000, 12000), (12000, 20000)]
    band_names = ['bass', 'mid', 'upper_mid', 'treble']
    
    raw_energy = {band: compute_band_energy(audio_raw, sr, low, high) 
                 for band, (low, high) in zip(band_names, bands)}
    ref_energy = {band: compute_band_energy(audio_ref, sr, low, high) 
                 for band, (low, high) in zip(band_names, bands)}

    for band in band_names:
        features.extend([
            np.array([raw_energy[band]]),
            np.array([ref_energy[band]]),
            np.array([ref_energy[band] - raw_energy[band]])
        ])

    for feature_type in feature_types:
        raw_feature = extract_features(audio_raw, sr, feature_type)
        ref_feature = extract_features(audio_ref, sr, feature_type)
        feature_diff = ref_feature - raw_feature

        features.append(np.mean(raw_feature, axis=1))
        features.append(np.std(raw_feature, axis=1)) 
        features.append(np.mean(ref_feature, axis=1)) 
        features.append(np.std(ref_feature, axis=1))
        features.append(np.mean(feature_diff, axis=1))

    feature_vector = np.concatenate([f.flatten() for f in features])
    return feature_vector


def extract_features_for_eq_prediction(audio, sr, frame_size=2048, hop_length=512):
    """
    Extracts features from raw audio to train a neural network for equalizer settings.

    Parameters:
      audio : raw audio signal
      sr : sampling rate in Hz
      frame_size : frame size for spectral analysis
      hop_length : number of samples between successive frames

    Returns:
      features : dictionary with all computed features
    """
    
    audio = audio / (np.max(np.abs(audio)) + 1e-10)
    
    bands = [(20, 600), (600, 6000), (6000, 12000), (12000, 20000)]
    band_names = ['bass', 'mid', 'upper_mid', 'treble']
    
    features = {}

    # 1. Frequency band energy analysis 
    # Calculate energy content in each frequency band
    # This helps understand the spectral balance of the audio
    band_energies = [] 
    for band_name, (low, high) in zip(band_names, bands):
        band_energy = compute_band_energy(audio, sr, low, high, n_fft=frame_size)
        features[f'energy_{band_name}'] = band_energy
        band_energies.append(band_energy)  

    total_band_energy = sum(band_energies) + 1e-8  
    for i, band_name in enumerate(band_names):
        features[f'energy_ratio_{band_name}'] = band_energies[i] / total_band_energy

    # 2. Frequency band relationship ratios 
    # These ratios capture important spectral balance characteristics
    # that directly relate to EQ adjustments needed
    features['bass_to_mid_ratio'] = band_energies[0] / (band_energies[1] + 1e-8)
    features['mid_to_upper_ratio'] = band_energies[1] / (band_energies[2] + 1e-8)
    features['upper_to_treble_ratio'] = band_energies[2] / (band_energies[3] + 1e-8)
    features['bass_to_treble_ratio'] = band_energies[0] / (band_energies[3] + 1e-8)

    # 3. Key spectral characteristics

    # Spectral centroid: "center of mass" of the spectrum (brightness indicator)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    features['spectral_centroid_mean'] = np.mean(centroid)
    features['spectral_centroid_std'] = np.std(centroid)

    # Spectral flatness: measure of how noise-like vs. tonal the signal is
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=frame_size, hop_length=hop_length)[0]
    features['spectral_flatness_mean'] = np.mean(flatness)

    # Spectral contrast: difference between peaks and valleys in spectrum
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=frame_size, hop_length=hop_length)
    for i in range(contrast.shape[0]):
        features[f'spectral_contrast_{i}'] = np.mean(contrast[i])

    # Spectral rolloff: frequency below which 85% of energy is contained
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=frame_size, hop_length=hop_length)[0]
    features['spectral_rolloff_mean'] = np.mean(rolloff)

    # 4. Spectral slope analysis
    # Spectral slope indicates the overall tonal balance (bright vs. dark)
    # Negative slopes suggest more high-frequency energy, positive slopes suggest more low-frequency energy
    S = np.abs(librosa.stft(audio, n_fft=frame_size, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    S_db = librosa.amplitude_to_db(S, ref=np.max)

    slopes = []
    for i in range(S_db.shape[1]):
        valid_freqs = freqs > 20
        if np.sum(valid_freqs) > 1:  
            x = np.log10(freqs[valid_freqs])
            y = S_db[valid_freqs, i]
            try:
                slope, _ = np.polyfit(x, y, 1)
                slopes.append(slope)
            except:
                pass
    
    features['spectral_slope_mean'] = np.mean(slopes) if slopes else 0
    features['spectral_slope_std'] = np.std(slopes) if slopes else 0
    
    # 5. Spectral sharpness
    # Measures the "sharpness" of the spectrum (high-frequency emphasis)
    # Higher values indicate more high-frequency content
    total_energy = np.sum(S) + 1e-10
    freq_weighted_sum = np.sum(S * freqs[:, np.newaxis]) / total_energy
    features['spectral_sharpness'] = freq_weighted_sum
    
    # 6. Spectral variance
    # Second moment of the spectrum - measures spread around the centroid
    # Higher values indicate wider spectral distribution
    S_sum = np.sum(S, axis=1) + 1e-10
    centroid_freq = np.sum(S_sum * freqs) / np.sum(S_sum)
    spectral_variance = np.sum(S_sum * (freqs - centroid_freq)**2) / np.sum(S_sum)
    features['spectral_variance'] = spectral_variance

    # 7. Spectral moments
    # Higher-order moments describe the shape of the frequency distribution
    mean_spectrum = np.mean(S, axis=1)
    normalized_spectrum = mean_spectrum / (np.sum(mean_spectrum) + 1e-10)

    freq_mean = np.sum(freqs * normalized_spectrum)
    freq_var = np.sum(((freqs - freq_mean) ** 2) * normalized_spectrum)
    freq_std = np.sqrt(freq_var + 1e-10)
    
    if freq_std > 0:
        # Skewness: asymmetry of the distribution (positive = tail extends toward higher frequencies)
        spectral_skewness = np.sum(((freqs - freq_mean) ** 3) * normalized_spectrum) / (freq_std ** 3)

        # Kurtosis: measure of tail heaviness (higher = more extreme values)
        spectral_kurtosis = np.sum(((freqs - freq_mean) ** 4) * normalized_spectrum) / (freq_std ** 4)
    else:
        spectral_skewness = 0
        spectral_kurtosis = 0
    
    features['spectral_skewness'] = spectral_skewness
    features['spectral_kurtosis'] = spectral_kurtosis

    # 8. Mel-spectrogram features
    # Mel scale is perceptually motivated and matches human auditory perception
    n_mels = 128
    mel_spec = librosa.feature.melspectrogram(
        y=audio, 
        sr=sr, 
        n_fft=frame_size,
        hop_length=hop_length,
        n_mels=n_mels
    )

    S_dB = librosa.power_to_db(mel_spec, ref=np.max)

    mel_bands = []
    for low_freq, high_freq in bands:
        low_mel = librosa.hz_to_mel(low_freq)
        high_mel = librosa.hz_to_mel(high_freq)
        max_mel = librosa.hz_to_mel(sr/2)
        low_idx = int(np.clip(low_mel / max_mel * (n_mels-1), 0, n_mels-1))
        high_idx = int(np.clip(high_mel / max_mel * (n_mels-1), 0, n_mels-1))
        high_idx = max(low_idx, high_idx)  
        mel_bands.append((low_idx, high_idx))

    # Extract statistical features from each mel frequency band
    for band_name, (low_idx, high_idx) in zip(band_names, mel_bands):
        if low_idx <= high_idx: 
            band_data = S_dB[low_idx:high_idx+1, :]
            if band_data.size > 0:  
                features[f'mel_{band_name}_mean'] = np.mean(band_data)
                features[f'mel_{band_name}_std'] = np.std(band_data)     
            else:
                features[f'mel_{band_name}_mean'] = 0
                features[f'mel_{band_name}_std'] = 0

    # 9. RMS energy characteristics
    # Root Mean Square energy provides information about the overall amplitude
    # and dynamic characteristics of the audio signal
    rms = librosa.feature.rms(y=audio, frame_length=frame_size, hop_length=hop_length)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_dynamic_range'] = np.max(rms) - np.min(rms)

    return features
