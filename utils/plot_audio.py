import numpy as np, librosa, matplotlib.pyplot as plt, soundfile as sf

def save_waveform_and_spec(path, out_wave="waveform.png", out_spec="spectrogram.png"):
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav[:,0]

    import matplotlib
    matplotlib.use("Agg")

    # waveform
    plt.figure(figsize=(12,3)); plt.plot(np.arange(len(wav))/sr, wav, linewidth=0.5)
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.title("Waveform")
    plt.tight_layout(); plt.savefig(out_wave, dpi=160); plt.close()

    # spectrogram
    S = np.abs(librosa.stft(wav, n_fft=1024, hop_length=256))**2
    import librosa.display
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(12,4)); librosa.display.specshow(S_db, sr=sr, hop_length=256, x_axis="time", y_axis="hz")
    plt.colorbar(format="%+2.0f dB"); plt.title("Spectrogram")
    plt.tight_layout(); plt.savefig(out_spec, dpi=160); plt.close()
