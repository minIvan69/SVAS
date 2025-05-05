import os
RABBIT_URL = os.getenv("RABBIT_URL")
DB_URL     = os.getenv("DB_URL")
MODEL_DIR  = os.getenv("MODEL_DIR", "/models/spkrec-ecapa-voxceleb")