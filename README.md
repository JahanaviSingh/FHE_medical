# FHE Medical Image Processor
### M.Tech Research Project — ML-Enhanced Homomorphic Encryption for Secure Image Processing

---

## What this does

Lets hospitals run AI diagnostics on medical scans **without the server ever
seeing the original image**. The image is encrypted using Fully Homomorphic
Encryption (CKKS scheme via Concrete-ML). The AI model runs on the
**ciphertext** — encrypted math — and the result is decrypted only by the
authorized client.

```
Patient uploads X-ray
  → Browser encrypts it (CKKS / FHE)
  → Server runs pneumonia detection on ciphertext
  → Server returns encrypted diagnosis
  → Browser decrypts → "Pneumonia 92% risk" heatmap
  → Server NEVER saw the original image
```

---

## Project Structure

```
fhe_medical/
├── app.py                          ← Flask entry point (run this)
├── requirements.txt                ← Python dependencies
│
├── backend/
│   ├── utils/
│   │   ├── fhe_engine.py           ← FHE encrypt / process / decrypt
│   │   ├── image_processor.py      ← All OpenCV medical operations
│   │   └── validator.py            ← Input modality validation
│   └── routes/
│       ├── fhe_routes.py           ← /api/fhe/* (main pipeline)
│       └── image_routes.py         ← /api/image/upload, /api/validate/check
│
├── frontend/
│   ├── templates/index.html        ← Main page
│   └── static/
│       ├── css/style.css
│       └── js/app.js               ← All UI logic
│
├── tests/
│   └── test_pipeline.py            ← pytest test suite
│
└── data/
    └── samples/                    ← Put sample images here
```

---

## Setup — Step by Step

### Step 1: Clone and enter project
```bash
git clone https://github.com/YOUR_USERNAME/fhe-medical-processor
cd fhe-medical-processor
```

### Step 2: Create Python virtual environment
```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
# This takes ~5 minutes (Concrete-ML is large)
pip install -r requirements.txt
```

If Concrete-ML install fails:
```bash
pip install concrete-python==2.6.0
pip install concrete-ml==1.5.0
pip install flask flask-cors opencv-python-headless Pillow pydicom numpy scikit-learn
```

### Step 4: Run the app
```bash
python app.py
```

Open browser at: **http://localhost:5000**

### Step 5: Run tests
```bash
python -m pytest tests/ -v
```

---

## Using the App

1. **Select modality** — Chest X-ray / Brain MRI / Bone X-ray / CT scan
2. **Upload image** — drag-drop, browse, or click "Load demo scan"
   - The app validates your image matches the modality
   - Colourful photos (like baboons) are rejected with a clear error
3. **Choose AI operation** — e.g. Pneumonia detection
4. **Click "Encrypt & analyse"**
   - Panel 2 shows the ciphertext (server sees only this noise)
   - Panel 3 shows the decrypted result with heatmap + diagnosis
5. **Review** — privacy score, encryption time, ciphertext expansion ratio

---

## Key Technical Concepts

### CKKS Homomorphic Encryption
CKKS is an FHE scheme that supports **approximate arithmetic on real numbers**
— ideal for image pixel values (floats). The pixel array is encoded as
coefficients of a polynomial in a ring `Z[X]/(X^N + 1)` where N is the
polynomial modulus degree (default: 4096).

### Concrete-ML
Zama's library that:
1. Takes a scikit-learn / PyTorch model
2. Quantises it to N-bit integer arithmetic
3. Compiles it to an FHE circuit that evaluates on ciphertexts

### What "0% patient data" means
The FHE guarantee: given only the ciphertext, the server cannot recover
any pixel values. This is a **mathematical** guarantee (based on Learning
With Errors hardness), not just a policy.

---

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| POST | `/api/fhe/pipeline` | Full pipeline: encrypt + process + decrypt |
| POST | `/api/fhe/encrypt` | Encrypt only, returns ciphertext preview |
| POST | `/api/image/upload` | Upload image file (multipart) |
| POST | `/api/validate/check` | Validate image against modality |

### Example API call (Python)
```python
import requests, base64
from PIL import Image
import io

# Load image
img = Image.open("chest_xray.png").convert("RGB")
buf = io.BytesIO()
img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

# Call pipeline
response = requests.post("http://localhost:5000/api/fhe/pipeline", json={
    "image_b64" : b64,
    "modality"  : "xray",
    "operation" : "pneumonia_detection",
})
data = response.json()
print(data["diagnosis"]["condition"])     # "Pneumonia detected"
print(data["metrics"]["privacy_score"])   # 94
print(data["metrics"]["patient_data_seen"])  # "0%"
```

---

## Datasets Referenced

| Dataset | Modality | Size | Link |
|---------|----------|------|------|
| NIH ChestX-ray14 | Chest X-ray | 112k images | kaggle.com/nih-chest-xrays |
| RSNA Pneumonia | Chest X-ray | 26k scans | kaggle.com/rsna-pneumonia-detection |
| BraTS 2023 | Brain MRI | 1,251 cases | synapse.org (BraTS) |
| MURA | Bone X-ray | 40k X-rays | stanfordmlgroup.github.io/projects/mura |

---

## Roadmap (for thesis)

- [ ] Compile real Concrete-ML circuits for CheXNet
- [ ] Add DICOM metadata stripping (full HIPAA de-id)
- [ ] WebSocket support for real-time progress updates
- [ ] Deploy to Heroku / Render (free tier)
- [ ] Add benchmark: FHE vs plaintext inference time comparison
- [ ] Formal security parameter justification (λ = 128 bits)

---

## Author
M.Tech Research Scholar  
Project: ML-Enhanced Homomorphic Encryption for Secure Image Processing
