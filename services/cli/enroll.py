"""
enroll.py  usage:
python enroll.py <user_id> wav1.wav wav2.wav ...
Сохраняет усреднённый эмбеддинг в БД.
"""
import sys, torch, librosa, psycopg
from speechbrain.pretrained import SpeakerRecognition

model = SpeakerRecognition.from_hparams(
    source="/models/spkrec-ecapa-voxceleb")
user, *files = sys.argv[1:]
vecs = []
for fn in files[:5]:
    wav, sr = librosa.load(fn, sr=16000)
    vec = model.encode_batch(torch.tensor(wav).unsqueeze(0))[0].squeeze()
    vecs.append(vec)
profile = torch.stack(vecs).mean(0).numpy()

with psycopg.connect("dbname=voiceid user=postgres password=postgres") as conn:
    conn.execute("INSERT INTO voices(user_id, embedding) VALUES (%s, %s)",
                 (user, profile.tolist()))
