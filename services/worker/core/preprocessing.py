import rnnoise
import torchaudio
def denoise_and_split(wav_path):
    wav, sr = torchaudio.load(wav_path)
    clean = rnnoise.reduce_noise(wav)
    segments = vad_split(clean, sr)
    return segments