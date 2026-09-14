import os
import scipy.io.wavfile as wv
from scipy.interpolate import interp1d
import numpy as np
import librosa
import time

import pytsmod as tsm
import pyworld as pw

ANONIMISERS = [('pytsmod', '2keyup'),
               ('pytsmod', '2keydown'),
               ('pytsmod', '3keyup'),
               ('pytsmod', '3keydown'),
               ('pytsmod', '4keyup'),
               ('pytsmod', '4keydown'),
               ('pytsmod', '5keyup'),
               ('pytsmod', '5keydown'),
               ('pytsmod', '6keyup'),
               ('pytsmod', '6keydown'),
               ('pyworldvocoder', '2keyup'),
               ('pyworldvocoder', '2keydown'),
               ('pyworldvocoder', '3keyup'),
               ('pyworldvocoder', '3keydown'),
               ('pyworldvocoder', '4keyup'),
               ('pyworldvocoder', '4keydown'),
               ('pyworldvocoder', '5keyup'),
               ('pyworldvocoder', '5keydown'),
               ('pyworldvocoder', '6keyup'),
               ('pyworldvocoder', '6keydown'),
               ('pyworldvocoder', 'formant15up'),
               ('pyworldvocoder', 'formant15down')]

COEFFS = {
        "2keyup": 2/12,
        "2keydown": -2/12,
        "3keyup": 3/12,
        "3keydown": -3/12,
        "4keyup": 4/12,
        "4keydown": -4/12,
        "5keyup": 5/12,
        "5keydown": -5/12,
        "6keyup": 6/12,
        "6keydown": -6/12,
        "formant15up": 1.15,
        "formant15down": 0.85                
    }


def makesavepaths(wavpath):
    path = os.path.normpath(wavpath)
    dirs = path.split(os.sep)
    dirname, n1, n2, wavname = dirs[0], dirs[-3], dirs[-2], dirs[-1]

    savepaths = []
    for method in ANONIMISERS:
        lib, params = method[0], method[1]
        new_wavname = f"{lib}_{params}_{wavname}"
        savepath = os.path.join('anon_LibriTTS', dirname, lib, params, n1, n2, new_wavname)
        savepaths.append( (savepath, lib, params) )

    return savepaths


def extract_f0_pytsmod(x, sr):
    f0, _, _ = librosa.pyin(x, fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'), sr=sr,
                            frame_length=800, hop_length=240, fill_na=None)

    return f0


def pitchshift_pytsmod(x, sr, f0, beta):

    x_new = tsm.tdpsola(x, sr, f0, beta=beta, p_hop_size=240, p_win_size=800)

    return x_new


def extract_params_pyworldvocoder(x, sr):

    f0, sp, ap = pw.wav2world(x, sr)

    return f0, sp, ap


def pitchshift_pyworldvocoder(sr, f0, sp, ap, beta):

    f0_new = f0 * beta

    x_new = pw.synthesize(f0_new, sp, ap, sr)

    return x_new


def formantshift_pyworldvocoder(sr, f0, sp, ap, alpha):

    n_freq_bins = sp.shape[1]
    original_freqs = np.linspace(0, sr / 2, n_freq_bins)
    warped_freqs = original_freqs * alpha
    warped_freqs = np.clip(warped_freqs, 0, sr / 2)

    sp_new = np.zeros_like(sp)
    for i in range(sp.shape[0]):
        f = interp1d(warped_freqs, sp[i, :], kind='linear', 
                     bounds_error=False, fill_value=(sp[i, 0], sp[i, -1]))
        sp_new[i, :] = f(original_freqs)

    x_new = pw.synthesize(f0, sp_new, ap, sr)
    
    return x_new


def process_and_save_audio(x, sr, f0_tsm, f0_pw, sp_pw, ap_pw, savepath, lib, params):
    coeff = COEFFS[params]

    if lib == 'pytsmod':
        x = pitchshift_pytsmod(x, sr, f0_tsm, pow(2, coeff))
    elif lib == 'pyworldvocoder':
        if params.startswith('formant'):
            x = formantshift_pyworldvocoder(sr, f0_pw, sp_pw, ap_pw, coeff)
        else:
            x = pitchshift_pyworldvocoder(sr, f0_pw, sp_pw, ap_pw, pow(2, coeff))
    else:
        raise ValueError(f"Unknown method {lib}")

    x = np.clip(x, -1.0, 1.0)
    x_int16 = (x * 32767).astype(np.int16)
    tmp_path = savepath + '.tmp'
    wv.write(tmp_path, 24000, x_int16)
    os.replace(tmp_path, savepath)


def is_valid_wav(filepath):
    try:
        sr, data = wv.read(filepath)
        return data.size > 0
    except Exception:
        return False


if __name__ == "__main__":

    print('Executing...')
    print("Deleting temporary files...")

    for root, _, files in os.walk('anon_LibriTTS'):
        for f in files:
            if f.endswith('.tmp'):
                os.remove(os.path.join(root, f))

    print("Temporary files deleted.")
    
    dirnames = ['dev-clean/LibriTTS/dev-clean',
                'dev-other/LibriTTS/dev-other',
                'test-clean/LibriTTS/test-clean',
                'test-other/LibriTTS/test-other']

    all_wavpaths = []

    for dirname in dirnames:
        idirs = [os.path.join(dirname, name) for name in os.listdir(dirname)]
        for idir in idirs:
            jdirs = [os.path.join(idir, name) for name in os.listdir(idir)]
            for jdir in jdirs:
                wavpaths = [os.path.join(jdir, name) for name in os.listdir(jdir) 
                           if name.endswith('.wav')]
                all_wavpaths.extend(wavpaths)

    total_wavpaths = len(all_wavpaths)

    print("Processing audiofiles...")

    start = time.time()

    for i, wavpath in enumerate(all_wavpaths, 1):
        
        sr, x = wv.read(wavpath)
        x = x.astype(np.float64) / 32768.0

        f0_tsm = extract_f0_pytsmod(x, sr)
        f0_pw, sp_pw, ap_pw = extract_params_pyworldvocoder(x, sr)

        savepaths = makesavepaths(wavpath)
        for s in savepaths:
            savepath, lib, params = s
            if os.path.exists(savepath) and is_valid_wav(savepath):
                continue
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            try:
                process_and_save_audio(x, sr, f0_tsm, f0_pw, sp_pw, ap_pw, savepath, lib, params)
            except Exception as e:
                print(f"ERROR: {wavpath} → {e}")
                continue
        elapsed = time.time() - start
        print(f"{i}/{total_wavpaths} ({round(100 * i / total_wavpaths, 2)}%) audiofiles processed.")
        print(f"Time elapsed: {round(elapsed / 60, 1)}m ({round(elapsed / 3600, 2)}h)")
        eta_m = (total_wavpaths - i) * elapsed / (i * 60)
        print(f"ETA: {round(eta_m, 1)}m ({round(eta_m / 60, 2)}h)")

    print("Processing finished.")