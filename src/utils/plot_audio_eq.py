import librosa
import numpy as np
import matplotlib.pyplot as plt

## Visualization of sound characteristics

def plot_eq_response(sr, params, fft_size=32768):
    """
    Visualizes the amplitude-frequency response of an equalizer

    Parameters:
      sr: sampling rate
      params: equalizer parameters
      fft_size: FFT size for calculating the frequency response

    Returns:
      fig: matplotlib graph object
    """

    from src.core.audio_equalizer import apply_equalizer

    test_signal = np.random.randn(fft_size)

    processed_signal = apply_equalizer(test_signal, sr, params, plot_response=False)

    orig_spectrum = np.abs(np.fft.rfft(test_signal))
    proc_spectrum = np.abs(np.fft.rfft(processed_signal))

    freq_response = 20 * np.log10(proc_spectrum / (orig_spectrum + 1e-8))

    freqs = np.fft.rfftfreq(fft_size, 1/sr)

    plt.figure(figsize=(12, 6))
    plt.semilogx(freqs, freq_response)
    plt.grid(True, which="both", ls="-", alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Equalizer Frequency Response')
    plt.xlim(20, sr / 2)
    plt.ylim(-30, 30)

    colors = ['r', 'g', 'b', 'purple']
    labels = ['Low Shelf', 'Peak Low', 'Peak High', 'High Shelf']
    freqs = [
        params['low_shelf_freq'], 
        params['peak_low_freq'], 
        params['peak_high_freq'], 
        params['high_shelf_freq']
    ]
    gains = [
        params['low_shelf_gain'], 
        params['peak_low_gain'], 
        params['peak_high_gain'], 
        params['high_shelf_gain']
    ]
    
    for i, (f, g, c, l) in enumerate(zip(freqs, gains, colors, labels)):
        plt.axvline(x=f, color=c, linestyle='--', alpha=0.7)
        plt.plot(f, g, 'o', color=c, markersize=8, label=f'{l}: {f}Hz, {g}dB')
    
    plt.legend()
    plt.tight_layout()
    return plt.gcf()


def plot_spectrograms(raw_audio, processed_audio, reference_audio, sr, title="Spectrograms Comparison"):
    """
    Visualizes and compares spectrograms of input, processed, and reference audio

    Parameters:
      raw_audio: raw audio
      processed_audio: processed audio
      reference_audio: reference audio
      sr: sample rate
      title: title for the plot

    Returns:
      fig: matplotlib plot object
    """
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    n_fft = 2048
    hop_length = 512

    S_raw = librosa.amplitude_to_db(
        np.abs(librosa.stft(raw_audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    librosa.display.specshow(
        S_raw, y_axis='log', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[0]
    )
    axes[0].set_title('Raw Audio')
    axes[0].label_outer()

    S_proc = librosa.amplitude_to_db(
        np.abs(librosa.stft(processed_audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    librosa.display.specshow(
        S_proc, y_axis='log', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[1]
    )
    axes[1].set_title('Processed Audio')
    axes[1].label_outer()

    S_ref = librosa.amplitude_to_db(
        np.abs(librosa.stft(reference_audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )
    librosa.display.specshow(
        S_ref, y_axis='log', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[2]
    )
    axes[2].set_title('Reference Audio')
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig
