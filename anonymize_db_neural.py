# from huggingface_hub import login
# login(token="use your token here")

import os
import numpy as np
import time
import scipy.io.wavfile as wv
import torch

from chatterbox.vc import ChatterboxVC

ANONIMISERS = [('chatterbox', 'voice1'), ('chatterbox', 'voice2')]

TARGET_VOICES = {
        "dev-clean": ("target_voices/dev-clean_voice1.wav", # 8297_275154_000006_000000.wav
                      "target_voices/dev-clean_voice2.wav"), # 8842_304647_000046_000000.wav
        "dev-other": ("target_voices/dev-other_voice1.wav", # 116_288045_000016_000001.wav
                      "target_voices/dev-other_voice2.wav"), # 700_122866_000002_000001.wav
        "test-clean": ("target_voices/test-clean_voice1.wav", # 260_123286_000014_000000.wav
                       "target_voices/test-clean_voice2.wav"), # 121_121726_000004_000003.wav
        "test-other": ("target_voices/test-other_voice1.wav", # 2609_156975_000026_000001.wav
                       "target_voices/test-other_voice2.wav") # 367_130732_000001_000002.wav
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
        savepaths.append( (savepath, params, dirname) )

    return savepaths
 

def process_and_save_audio(filepath, targetvoice_path, model, savepath):

    wav = model.generate(
            audio=filepath,
            target_voice_path=targetvoice_path,
        )

    x_np = wav.cpu().numpy().squeeze()
    x_np = np.clip(x_np, -1.0, 1.0)
    x_int16 = (x_np * 32767).astype(np.int16)

    tmp_path = savepath + '.tmp'
    wv.write(tmp_path, model.sr, x_int16)
    os.replace(tmp_path, savepath)


def is_valid_wav(filepath):
    return os.path.getsize(filepath) > 64


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

    print("Choosing device for Chatterbox...")

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Chatterbox is using: {device}")
    print("Initializing Chatterbox model...")

    model = ChatterboxVC.from_pretrained(device)

    print("Chatterbox model initialized.")
    print("Processing audiofiles...")

    start = time.time()

    for i, wavpath in enumerate(all_wavpaths, 1):
        savepaths = makesavepaths(wavpath)
        for s in savepaths:
            savepath, params, dirname = s
            if os.path.exists(savepath) and is_valid_wav(savepath):
                continue
            os.makedirs(os.path.dirname(savepath), exist_ok=True)

            if params == "voice1":
                targetvoice_path = TARGET_VOICES[dirname][0]
            elif params == "voice2":
                targetvoice_path = TARGET_VOICES[dirname][1]
            else:
                raise ValueError(f"Unknown parameter {params}")

            try:
                process_and_save_audio(wavpath, targetvoice_path, model, savepath)
            except Exception as e:
                print(f"ERROR: {wavpath} → {e}")
                continue
        elapsed = time.time() - start
        print(f"{i}/{total_wavpaths} ({round(100 * i / total_wavpaths, 2)}%) audiofiles processed.")
        print(f"Time elapsed: {round(elapsed / 60, 1)}m ({round(elapsed / 3600, 2)}h)")
        eta_m = (total_wavpaths - i) * elapsed / (i * 60)
        print(f"ETA: {round(eta_m, 1)}m ({round(eta_m / 60, 2)}h)")

    print("Processing finished.")