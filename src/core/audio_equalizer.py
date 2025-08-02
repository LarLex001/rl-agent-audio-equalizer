import numpy as np
import scipy.signal as signal
from src.utils.plot_audio_eq import plot_eq_response

# Equalizer and equalizer filters

def biquad_peaking_filter(audio, sr, freq, gain, q):
    """
    Implements a second-order peak filter (biquad)

    Parameters:
        audio: input audio signal
        sr: sampling rate
        freq: filter center frequency (Hz)
        gain: signal gain/attenuation (dB)
        q: filter quality

    Returns:
        filtered: filtered audio signal
    """

    freq = max(20, min(freq, sr / 2 * 0.95))
    q = max(0.1, min(q, 18))  

    A = 10**(gain / 40.0)
    w0 = 2 * np.pi * freq / sr

    alpha = np.sin(w0) / (2 * q)

    b0 = 1 + alpha * A
    b1 = -2 * np.cos(w0)
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha / A

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])

    filtered = signal.filtfilt(b, a, audio)

    if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
        print(f"Unstable filter: freq={freq:.1f}Hz, gain={gain:.1f}dB, q={q:.1f}")
        return audio
    
    return filtered


def shelf_filter(audio, sr, freq, gain, q, filter_type="low"):
    """
    Implements a shelf filter (low or high frequencies)

    Parameters:
        audio: input audio signal
        sr: sampling frequency
        freq: cutoff frequency (Hz)
        gain: signal gain/attenuation (dB)
        q: filter quality
        filter_type: filter type ("low" - low frequencies, "high" - high frequencies)

    Returns:
        filtered: filtered audio signal
    """
    
    freq = max(20, min(freq, sr / 2 * 0.95))
    q = max(0.1, min(q, 10.0)) # slope instead of Q for shelf filters
    
    A = 10**(gain / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    
    if filter_type == "low":
        # Low-shelf
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    else:
        # High-shelf
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1/a0, a2/a0])
    
    filtered = signal.filtfilt(b, a, audio)

    if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
        print(f"Unstable {filter_type}-shelf filter: freq={freq:.1f}Hz, gain={gain:.1f}dB")
        return audio
    
    return filtered


def apply_equalizer(audio, sr, params, plot_response=False):
    """
    Applies all four filters to audio with given parameters

    Parameters:
        audio: audio signal (numpy array)
        sr: sampling rate (Hz)
        params: dictionary with parameters for all filters
        plot_response: if True, displays frequency response

    Returns:
        processed_audio: processed audio signal
    """
    
    processed_audio = audio.copy()

    # application of filters
    processed_audio = shelf_filter(
        processed_audio, sr,
        params['low_shelf_freq'], 
        params['low_shelf_gain'], 
        params['low_shelf_q'], 
        filter_type="low"
    )

    processed_audio = biquad_peaking_filter(
        processed_audio, sr,
        params['peak_low_freq'],
        params['peak_low_gain'],
        params['peak_low_q']
    )

    processed_audio = biquad_peaking_filter(
        processed_audio, sr,
        params['peak_high_freq'],
        params['peak_high_gain'],
        params['peak_high_q']
    )

    processed_audio = shelf_filter(
        processed_audio, sr,
        params['high_shelf_freq'],
        params['high_shelf_gain'],
        params['high_shelf_q'],
        filter_type="high"
    )

    if plot_response:
        plot_eq_response(sr, params)
    
    return processed_audio
