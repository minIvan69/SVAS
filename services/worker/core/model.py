from speechbrain.pretrained import SpeakerRecognition
from .config import MODEL_DIR

_model = None
def get_model():
    global _model
    if _model is None:
        _model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=MODEL_DIR
        )
    return _model

# обёртка, возвращающая готовый вектор или батч векторов