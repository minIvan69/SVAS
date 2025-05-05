"""
verify.py  usage:
python verify.py <user_id> test.wav
Выводит cosine‑score и "True/False".
"""
import sys, torch, librosa, psycopg, numpy as np
from speechbrain.pretrained import SpeakerRecognition

model = SpeakerRecognition.from_hparams(
    source="/models/spkrec-ecapa-voxceleb")
user, test = sys.argv[1:]
wav, sr = librosa.load(test, sr=16000)
test_vec = model.encode_batch(torch.tensor(wav).unsqueeze(0))[0].squeeze()

with psycopg.connect("dbname=voiceid user=postgres password=postgres") as conn:
    emb = conn.execute("SELECT embedding FROM voices WHERE user_id=%s",
                       (user,)).fetchone()
profile = np.array(emb[0])
score = float(torch.nn.functional.cosine_similarity(
              torch.tensor(test_vec), torch.tensor(profile), dim=0))
print("score:", score)
print("match:", score > 0.65)      # базовый порог
