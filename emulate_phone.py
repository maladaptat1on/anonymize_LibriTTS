import os
import time
import subprocess
import tempfile
import scipy.io.wavfile as wv
import numpy as np

from scipy.signal import resample_poly

CODECS = ['mulaw', 'alaw', 'amrnb', 'opus']


def equalize_level(x):

    target_dbfs = -26.0
    
    rms = np.sqrt(np.mean(x ** 2))

    if rms < 1e-9:
        return x
    
    current_dbfs = 20 * np.log10(rms)
    gain_db = target_dbfs - current_dbfs
    gain_linear = 10 ** (gain_db / 20)
    
    x_eq = x * gain_linear
    
    return x_eq


def resample(sr, x):

    target_sr = 8000
    
    if sr == target_sr:
        return sr, x

    x_res = resample_poly(x, 1, 3)
    
    return target_sr, x_res


def apply_codec(sr, x, codecs, bitrate='12.2k'):
    
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
        input_path = f_in.name
    wv.write(input_path, sr, x)

    xs_codec = []

    try:

        for codec in codecs:

            encoded_path = input_path + f'.{codec}.encoded'
            decoded_path = input_path + f'.{codec}.decoded.wav'   

            if codec == 'mulaw':
                encoded_path = encoded_path + '.wav'
                encode_cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-ar', '8000', '-ac', '1',
                    '-c:a', 'pcm_mulaw',
                    encoded_path
                ]

            elif codec == 'alaw':
                encoded_path = encoded_path + '.wav'
                encode_cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-ar', '8000', '-ac', '1',
                    '-c:a', 'pcm_alaw',
                    encoded_path
                ]

            elif codec == 'amrnb':
                encoded_path = encoded_path + '.amr'
                encode_cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-ar', '8000', '-ac', '1',
                    '-c:a', 'libopencore_amrnb',
                    '-b:a', bitrate,
                    encoded_path
                ]

            elif codec == 'opus':
                encoded_path = encoded_path + '.opus'
                encode_cmd = [
                    'ffmpeg', '-y', '-i', input_path,
                    '-ar', '8000', '-ac', '1',
                    '-c:a', 'libopus',
                    '-b:a', bitrate,
                    '-application', 'voip',
                    encoded_path
                ]

            else:
                raise ValueError(f"Unknown codec: {codec}")

            subprocess.run(encode_cmd, check=True, capture_output=True)

            decode_cmd = [
                'ffmpeg', '-y', '-i', encoded_path,
                '-ar', '8000', '-ac', '1',
                '-c:a', 'pcm_s16le',
                decoded_path
            ]
            subprocess.run(decode_cmd, check=True, capture_output=True)

            _, x_decoded = wv.read(decoded_path)
            xs_codec.append( (codec, x_decoded) )

            for path in [encoded_path, decoded_path]:
                if os.path.exists(path):
                    os.remove(path)

    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
    
    return xs_codec


def packet_loss(sr, x, loss_rate=0.02, packet_ms=20):

    packet_size = int(sr * packet_ms / 1000)
    
    n_packets = len(x) // packet_size
    
    x_lp = x.copy()
    
    for i in range(n_packets):

        if np.random.rand() < loss_rate:
            start = i * packet_size
            end = start + packet_size
                        
            if i > 0:
                x_lp[start:end] = x_lp[start - packet_size:start]
            else:
                x_lp[start:end] = 0
    
    return x_lp


if __name__ == "__main__":

    print("Executing...")
    
    filepaths = []
    for root, dirs, files in os.walk('anon_LibriTTS'):
        for filename in files:
            if filename.endswith('.wav'):
                filepaths.append(os.path.join(root, filename))

    total_paths = len(filepaths)

    print("Processing audiofiles...")

    start = time.time()

    for i, filepath in enumerate(filepaths, 1):

        sr, x = wv.read(filepath)
        x = x.astype(np.float64) / 32768.0

        x_equalized = equalize_level(x)
        sr_resampled, x_resampled = resample(sr, x_equalized)

        x_resampled = np.clip(x_resampled, -1.0, 1.0)
        x_resampled_int16 = (x_resampled * 32767).astype(np.int16)

        xs_codec = apply_codec(sr_resampled, x_resampled_int16, CODECS)

        xs_lostpackets = []

        for c, x_codec in xs_codec:
            x_lostpackets = packet_loss(sr_resampled, x_codec)
            xs_lostpackets.append( (c, x_lostpackets) )

        for c, x in xs_lostpackets:
            savepath = c + '_' + filepath
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            tmp_path = savepath + '.tmp'
            wv.write(tmp_path, sr_resampled, x)
            os.replace(tmp_path, savepath)

        elapsed = time.time() - start
        print(f"{i}/{total_paths} ({round(100 * i / total_paths, 2)}%) audiofiles processed.")
        print(f"Time elapsed: {round(elapsed / 60, 1)}m ({round(elapsed / 3600, 2)}h)")
        eta_m = (total_paths - i) * elapsed / (i * 60)
        print(f"ETA: {round(eta_m, 1)}m ({round(eta_m / 60, 2)}h)")
