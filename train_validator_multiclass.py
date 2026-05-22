"""
train_validator_multiclass.py  — PATCHED
Trains 5-class Random Forest: non_medical / xray / mri / bone / ct

Changes vs original:
  1. make_mri() now generates 30% composite 2×2 / 2×1 panel images
     so the model sees the feature profile that multi-slice viewers produce.
  2. Sanity check added for composite MRI.
  3. MRI class count raised to n*1.5 (class weighted too).

Run: python train_validator_multiclass.py
Output: backend/utils/scan_classifier_multiclass.pkl
"""

import numpy as np
import cv2
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

CLASSES = {0:"non_medical", 1:"xray", 2:"mri", 3:"bone", 4:"ct"}
MODALITY_ML_THRESHOLDS = {"xray":0.40, "mri":0.25, "bone":0.10, "ct":0.45}
# ↑ MRI threshold lowered 0.35→0.25 to account for composite-image score

FEATURE_NAMES = [
    "chroma","sat","colour_frac","edge_density","lap_var",
    "int_range","mean_int","std_int","hist_entropy","hist_max",
    "green_dom","red_dom","hf_energy","local_contrast","blob_diff",
    "grad_mean","cb_ratio","dark_frac","bright_frac","peaks64","std_norm",
]


# ─────────────────────────────────────────────
#  FEATURE EXTRACTOR  (unchanged — must stay identical to validator.py)
# ─────────────────────────────────────────────

def extract_features(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (128,128), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    r=img[:,:,2].astype(float); g=img[:,:,1].astype(float); b=img[:,:,0].astype(float)
    gray=(r*0.299+g*0.587+b*0.114).astype(np.uint8)
    chroma=abs(r.mean()-g.mean())+abs(g.mean()-b.mean())+abs(r.mean()-b.mean())
    max_c=np.maximum(np.maximum(r,g),b); min_c=np.minimum(np.minimum(r,g),b)
    sat=(max_c-min_c)/(max_c+1e-6)
    colour_frac=float(((max_c-min_c)>30).mean())
    edges=cv2.Canny(gray,40,120)
    edge_density=float(edges.sum())/(255.0*128*128)
    lap=cv2.Laplacian(gray,cv2.CV_64F); lap_var=float(lap.var())
    int_range=int(gray.max())-int(gray.min())
    hist,_=np.histogram(gray.flatten(),bins=32,range=(0,255))
    hn=hist.astype(float)/(hist.sum()+1e-6)
    hist_entropy=float(-np.sum(hn*np.log(hn+1e-10))); hist_max=float(hn.max())
    green_dom=float(((g>r+10)&(g>b+10)).mean())
    red_dom=float(((r>g+15)&(r>b+15)).mean())
    gray_f=gray.astype(np.float32); dct=cv2.dct(gray_f)
    total_e=(dct**2).sum()+1e-6; h2,w2=dct.shape
    hf_energy=float(((dct[h2//4:,w2//4:])**2).sum()/total_e)
    kernel=np.ones((8,8),np.float32)/64
    local_contrast=float(cv2.filter2D(gray.astype(float),-1,kernel).std())
    blurred=cv2.GaussianBlur(gray,(15,15),5)
    blob_diff=float(np.abs(gray.astype(float)-blurred.astype(float)).mean())
    gx=cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)
    grad_mean=float(np.sqrt(gx**2+gy**2).mean())
    h,w=gray.shape; margin=min(h,w)//4; cy,cx=h//2,w//2
    center=gray[cy-margin:cy+margin,cx-margin:cx+margin]
    border=np.concatenate([gray[:margin,:].flatten(),gray[-margin:,:].flatten(),
                            gray[:,:margin].flatten(),gray[:,-margin:].flatten()])
    cb_ratio=float(center.mean()+1)/(float(border.mean())+1)
    dark_frac=float((gray<20).mean())
    bright_frac=float((gray>180).mean())
    hist64,_=np.histogram(gray.flatten(),bins=64,range=(0,255))
    h64=hist64.astype(float)/(hist64.sum()+1e-6)
    thr64=h64.max()*0.10; peaks64=0; in_p=False
    for v in h64:
        if v>thr64 and not in_p: peaks64+=1; in_p=True
        elif v<=thr64: in_p=False
    return np.array([
        chroma, float(sat.mean()), colour_frac, edge_density, lap_var,
        int_range, float(gray.mean()), float(gray.std()), hist_entropy, hist_max,
        green_dom, red_dom, hf_energy, local_contrast, blob_diff,
        grad_mean, cb_ratio, dark_frac, bright_frac, float(peaks64),
        float(gray.std()/128)
    ], dtype=np.float32)


# ─────────────────────────────────────────────
#  AUGMENTATION  (unchanged)
# ─────────────────────────────────────────────

def augment(img, rng):
    img = np.clip(img.astype(float)*rng.uniform(0.6,1.5), 0, 255).astype(np.uint8)
    mean = img.mean()
    img = np.clip((img.astype(float)-mean)*rng.uniform(0.6,1.7)+mean, 0, 255).astype(np.uint8)
    if rng.random() > 0.5: img = cv2.flip(img, 1)
    if rng.random() > 0.5: img = cv2.flip(img, 0)
    M = cv2.getRotationMatrix2D((64,64), rng.uniform(-20,20), 1.0)
    img = cv2.warpAffine(img, M, (128,128))
    if rng.random() > 0.4:
        img = cv2.GaussianBlur(img, (0,0), rng.uniform(0.5,1.8))
    noise = rng.normal(0, rng.uniform(2,15), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    if rng.random() > 0.6:
        m = int(rng.uniform(3,14))
        img = cv2.resize(img[m:-m,m:-m], (128,128))
    return img


# ─────────────────────────────────────────────
#  CLASS 1: CHEST X-RAY  (unchanged)
# ─────────────────────────────────────────────

def make_xray(rng):
    size = 128
    img = np.zeros((size,size,3), dtype=np.uint8)
    img[:,:] = int(rng.integers(10, 30))
    cx = size//2 + int(rng.integers(-8,8))
    cy = size//2 + int(rng.integers(-5,5))
    for y in range(size):
        for x in range(size):
            dx=(x-cx)/(size*rng.uniform(0.33,0.43))
            dy=(y-cy)/(size*rng.uniform(0.39,0.49))
            d2=dx*dx+dy*dy
            if d2 < 1:
                v = int(40 + rng.uniform(70,120)*(1-d2)**rng.uniform(0.7,1.2))
                img[y,x] = [max(0,min(255,v))]*3
    n_ribs = int(rng.integers(4,8))
    for i in range(n_ribs):
        yr = int(size*0.18 + i*(size*0.62/n_ribs) + rng.integers(-3,3))
        thick = int(rng.integers(2,5))
        for yy in range(max(0,yr-thick), min(size,yr+thick)):
            for x in range(int(size*0.12), int(size*0.88)):
                v = min(255, int(img[yy,x,0]) + int(rng.integers(40,90)))
                img[yy,x] = [v,v,v]
    sx = size//2 + int(rng.integers(-3,3))
    for y in range(int(size*0.1), int(size*0.88)):
        v = min(255, int(img[y,sx,0]) + int(rng.integers(60,110)))
        for x in range(max(0,sx-3), min(size,sx+4)):
            img[y,x] = [v,v,v]
    sigma = rng.uniform(2.5, 5.0)
    for c in range(3):
        img[:,:,c] = cv2.GaussianBlur(img[:,:,c], (0,0), sigma)
    return augment(img, rng)


# ─────────────────────────────────────────────
#  CLASS 2: BRAIN MRI  (PATCHED — adds composite panels)
# ─────────────────────────────────────────────

def _make_single_mri_slice(rng, size=128):
    """One brain MRI slice (the original make_mri logic)."""
    img = np.zeros((size,size,3), dtype=np.uint8)
    img[:,:] = int(rng.integers(0,4))
    cx = size//2 + int(rng.integers(-8,8))
    cy = size//2 + int(rng.integers(-6,6))
    rx = rng.uniform(0.32,0.40); ry = rng.uniform(0.34,0.42)
    for y in range(size):
        for x in range(size):
            dx=(x-cx)/(size*rx); dy=(y-cy)/(size*ry); d2=dx*dx+dy*dy
            if d2 < 1:
                if d2 < 0.22:   v = int(rng.uniform(100,165))
                elif d2 < 0.52: v = int(rng.uniform(60,115))
                elif d2 < 0.80: v = int(rng.uniform(25,130))
                else:            v = int(rng.uniform(10,60))
                img[y,x] = [max(0,min(255,v))]*3
    for y in range(size):
        for x in range(size):
            dx=(x-cx)/(size*rx); dy=(y-cy)/(size*ry)
            r = np.sqrt(dx*dx+dy*dy)
            if 0.82 < r < 0.98:
                img[y,x] = [max(0,min(255,int(rng.uniform(30,90))))]*3
            elif 0.78 < r <= 0.82:
                img[y,x] = [max(0,min(255,int(rng.uniform(80,150))))]*3
    for _ in range(int(rng.integers(60,100))):
        angle = rng.uniform(0, 2*np.pi)
        r_t = rng.uniform(0.50, 0.82)
        px = int(cx + size*rx*r_t*np.cos(angle))
        py = int(cy + size*ry*r_t*np.sin(angle))
        pr = int(rng.integers(1,4)); bright = rng.random() > 0.5
        for y in range(max(0,py-pr), min(size,py+pr)):
            for x in range(max(0,px-pr), min(size,px+pr)):
                if (x-px)**2+(y-py)**2 <= pr**2:
                    dx2=(x-cx)/(size*rx); dy2=(y-cy)/(size*ry)
                    if dx2*dx2+dy2*dy2 < 1:
                        cur = int(img[y,x,0]); delta = int(rng.integers(45,90))
                        v = min(255,cur+delta) if bright else max(0,cur-delta)
                        img[y,x] = [v,v,v]
    vx=cx+int(rng.integers(-5,5)); vy=cy+int(rng.integers(-3,3)); vr=int(rng.integers(5,11))
    for y in range(max(0,vy-vr), min(size,vy+vr)):
        for x in range(max(0,vx-vr), min(size,vx+vr)):
            if (x-vx)**2+(y-vy)**2 < vr**2:
                img[y,x] = [max(0,min(255,int(rng.uniform(165,240))))]*3
    sigma = rng.uniform(0.2, 0.7)
    for c in range(3):
        img[:,:,c] = cv2.GaussianBlur(img[:,:,c], (0,0), sigma)
    if rng.random() > 0.40:
        blue_b = int(rng.uniform(10,55)); blue_g = int(rng.uniform(5,25))
        img[:,:,0] = np.clip(img[:,:,0].astype(np.int16)+blue_b, 0, 255).astype(np.uint8)
        img[:,:,1] = np.clip(img[:,:,1].astype(np.int16)+blue_g, 0, 255).astype(np.uint8)
        img[:,:,2] = np.clip(img[:,:,2].astype(np.int16)-int(rng.uniform(0,15)), 0, 255).astype(np.uint8)
    return img


def _make_composite_mri(rng, layout="2x2"):
    """
    NEW: composite multi-panel brain MRI (as produced by PACS/clinical viewers).
    Layout "2x2" = 4 slices in a 2×2 grid.
    Layout "2x1" = 2 slices side by side.

    Key features of a composite:
      - cb_ratio ~1.0  (no single bright centre — multiple brains)
      - dark_frac high (lots of black dividing lines + black FOV corners)
      - Multiple intensity peaks
    """
    size = 128
    half = size // 2

    if layout == "2x2":
        slices = [_make_single_mri_slice(rng, half) for _ in range(4)]
        panel  = np.zeros((size, size, 3), dtype=np.uint8)
        panel[:half, :half]   = slices[0]
        panel[:half, half:]   = slices[1]
        panel[half:, :half]   = slices[2]
        panel[half:, half:]   = slices[3]
        # Thin dark dividing lines (typical PACS grid)
        divider_val = int(rng.integers(0, 10))
        panel[half-1:half+1, :] = divider_val
        panel[:, half-1:half+1] = divider_val
    else:  # 2x1
        slices = [_make_single_mri_slice(rng, size) for _ in range(2)]
        s0 = cv2.resize(slices[0], (half, size))
        s1 = cv2.resize(slices[1], (half, size))
        panel = np.concatenate([s0, s1], axis=1)
        divider_val = int(rng.integers(0, 10))
        panel[:, half-1:half+1] = divider_val

    return augment(panel, rng)


def make_mri(rng):
    """
    PATCHED: 70% single slice, 30% composite panel.
    """
    if rng.random() < 0.30:
        layout = rng.choice(["2x2", "2x1"])
        return _make_composite_mri(rng, layout)
    return augment(_make_single_mri_slice(rng), rng)


# ─────────────────────────────────────────────
#  CLASS 3: BONE X-RAY  (unchanged)
# ─────────────────────────────────────────────

def make_bone(rng):
    size = 128
    img = np.zeros((size,size,3), dtype=np.uint8)
    img[:,:] = 0
    bt = rng.choice(["long","long","long","joint","wrist"])
    if bt == "long":
        bx = int(size * rng.uniform(0.20, 0.80))
        bw = int(size * rng.uniform(0.06, 0.18))
        st_width = int(size * rng.uniform(0.30, 0.50))
        for y in range(size):
            for x in range(max(0,bx-st_width), min(size,bx+st_width)):
                img[y,x] = [int(rng.uniform(25,85))]*3
        for y in range(int(size*0.02), int(size*0.98)):
            for x in range(size):
                d = abs(x - bx)
                if d < bw:
                    if d < bw*0.18 or d > bw*0.82:
                        v = int(rng.uniform(210,255))
                    else:
                        v = int(rng.uniform(130,200))
                    img[y,x] = [v,v,v]
        if rng.random() > 0.60:
            fy = int(rng.uniform(size*0.25, size*0.75))
            fangle = rng.uniform(-0.6, 0.6)
            for dx in range(-bw, bw):
                dy = int(dx * np.tan(fangle))
                ffy = fy + dy
                if 0 <= ffy < size and 0 <= bx+dx < size:
                    img[ffy, bx+dx] = [max(0, int(img[ffy,bx+dx,0])-60)]*3
    else:
        cx = size//2 + int(rng.integers(-10,10))
        cy = size//2 + int(rng.integers(-10,10))
        rx = int(size*rng.uniform(0.20,0.40)); ry = int(size*rng.uniform(0.18,0.35))
        for y in range(size):
            for x in range(size):
                dx=(x-cx)/rx; dy=(y-cy)/ry; d2=dx*dx+dy*dy
                if d2 < 1:
                    v = int(rng.uniform(170,255) * (0.5 + 0.5*(1-d2)**0.3))
                    img[y,x] = [max(0,min(255,v))]*3
        for y in range(size):
            for x in range(size):
                dx=(x-cx)/rx; dy=(y-cy)/ry; d2=dx*dx+dy*dy
                if 0.7 < d2 < 1.2:
                    v = int(rng.uniform(200,255))
                    img[y,x] = [v,v,v]
    sigma = rng.uniform(0.8, 2.5)
    for c in range(3):
        img[:,:,c] = cv2.GaussianBlur(img[:,:,c], (0,0), sigma)
    if rng.random() > 0.55:
        blue_b = int(rng.uniform(5,35))
        img[:,:,0] = np.clip(img[:,:,0].astype(np.int16)+blue_b, 0,255).astype(np.uint8)
    return augment(img, rng)


# ─────────────────────────────────────────────
#  CLASS 4: CT SCAN  (unchanged)
# ─────────────────────────────────────────────

def make_ct(rng):
    size = 128
    img = np.zeros((size,size,3), dtype=np.uint8)
    cx = size//2+int(rng.integers(-8,8)); cy = size//2+int(rng.integers(-8,8))
    rx = rng.uniform(0.35,0.46); ry = rng.uniform(0.37,0.48)
    for y in range(size):
        for x in range(size):
            dx=(x-cx)/(size*rx); dy=(y-cy)/(size*ry); d2=dx*dx+dy*dy
            if d2 < 1:
                if d2 < 0.12:   v = int(rng.uniform(40,90))
                elif d2 < 0.50: v = int(rng.uniform(80,160))
                elif d2 < 0.80: v = int(rng.uniform(50,120))
                else:            v = int(rng.uniform(180,240))
                img[y,x] = [max(0,min(255,v))]*3
    for _ in range(int(rng.integers(20,45))):
        angle = rng.uniform(0, 2*np.pi)
        r_t = rng.uniform(0.55,0.88)
        px = int(cx+size*rx*r_t*np.cos(angle)); py = int(cy+size*ry*r_t*np.sin(angle))
        pr = int(rng.integers(1,5))
        for y in range(max(0,py-pr), min(size,py+pr)):
            for x in range(max(0,px-pr), min(size,px+pr)):
                if (x-px)**2+(y-py)**2 < pr**2:
                    dx2=(x-cx)/(size*rx); dy2=(y-cy)/(size*ry)
                    if dx2*dx2+dy2*dy2 < 1:
                        img[y,x] = [max(0,min(255,int(rng.uniform(190,255))))]*3
    sigma = rng.uniform(0.8, 2.5)
    for c in range(3):
        img[:,:,c] = cv2.GaussianBlur(img[:,:,c], (0,0), sigma)
    if rng.random() > 0.35:
        blue_b=int(rng.uniform(15,80)); blue_g=int(rng.uniform(5,30))
        img[:,:,0]=np.clip(img[:,:,0].astype(np.int16)+blue_b,0,255).astype(np.uint8)
        img[:,:,1]=np.clip(img[:,:,1].astype(np.int16)+blue_g,0,255).astype(np.uint8)
        img[:,:,2]=np.clip(img[:,:,2].astype(np.int16)-int(rng.uniform(0,20)),0,255).astype(np.uint8)
    return augment(img, rng)


# ─────────────────────────────────────────────
#  CLASS 0: NON-MEDICAL  (unchanged)
# ─────────────────────────────────────────────

def make_nonmedical(rng):
    size = 128
    img = np.zeros((size,size,3), dtype=np.uint8)
    t = rng.choice(["colour_photo","screenshot","nature","animal","document",
                     "colour_photo","nature"])
    if t == "colour_photo":
        hue = int(rng.integers(0,180))
        for y in range(size):
            for x in range(size):
                h=int((hue+rng.integers(-25,25))%180)
                s=int(rng.integers(80,255)); v=int(rng.integers(60,255))
                bgr=cv2.cvtColor(np.array([[[h,s,v]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0]
                img[y,x]=bgr
        img=np.clip(img.astype(np.int16)+rng.integers(-40,40,(size,size,3)).astype(np.int16),0,255).astype(np.uint8)
    elif t == "screenshot":
        img[:,:] = int(rng.integers(230,255))
        for _ in range(int(rng.integers(3,9))):
            x1,y1=int(rng.integers(0,size-10)),int(rng.integers(0,size-10))
            x2=min(size-1,x1+int(rng.integers(15,60)))
            y2=min(size-1,y1+int(rng.integers(8,25)))
            cv2.rectangle(img,(x1,y1),(x2,y2),[int(c) for c in rng.integers(30,200,3)],-1)
        for _ in range(int(rng.integers(10,25))):
            y=int(rng.integers(5,size-5))
            x1=int(rng.integers(5,25)); x2=int(rng.integers(size//2,size-5))
            c=int(rng.integers(0,60))
            cv2.line(img,(x1,y),(x2,y),(c,c,c),1)
    elif t == "nature":
        for y in range(size):
            for x in range(size):
                gv=int(rng.integers(70,190))
                img[y,x]=[int(gv*rng.uniform(0.1,0.5)),gv,int(gv*rng.uniform(0.2,0.7))]
        img=np.clip(img.astype(np.int16)+rng.integers(-50,50,(size,size,3)).astype(np.int16),0,255).astype(np.uint8)
    elif t == "animal":
        hue=int(rng.integers(0,30))
        for y in range(size):
            for x in range(size):
                h=int((hue+rng.integers(-15,15))%180)
                s=int(rng.integers(60,180)); v=int(rng.integers(50,180))
                bgr=cv2.cvtColor(np.array([[[h,s,v]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0]
                img[y,x]=bgr
        for _ in range(4):
            img=np.clip(img.astype(np.int16)+rng.integers(-50,50,(size,size,3)).astype(np.int16),0,255).astype(np.uint8)
    elif t == "document":
        img[:,:] = int(rng.integers(235,255))
        for _ in range(int(rng.integers(12,28))):
            y=int(rng.integers(5,size-5))
            x1=int(rng.integers(5,20)); x2=int(rng.integers(size//2,size-5))
            c=int(rng.integers(0,50))
            cv2.line(img,(x1,y),(x2,y),(c,c,c),1)
    return augment(img, rng)


# ─────────────────────────────────────────────
#  BUILD DATASET  (PATCHED: more MRI samples)
# ─────────────────────────────────────────────

def build_dataset(n=600):
    print("Generating training data...")
    rng = np.random.default_rng(42)
    X, y = [], []
    generators = [
        (0, "Non-medical",  make_nonmedical, n*2),
        (1, "Chest X-ray",  make_xray,       n),
        (2, "Brain MRI",    make_mri,         int(n*1.5)),   # PATCH: more MRI samples
        (3, "Bone X-ray",   make_bone,        n),
        (4, "CT scan",      make_ct,          n),
    ]
    for cls, name, fn, count in generators:
        for _ in range(count):
            X.append(extract_features(fn(rng)))
            y.append(cls)
        print(f"  Class {cls} ({name}): {count}")
    print(f"  Total: {len(y)}")
    return np.array(X), np.array(y)


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

def train():
    X, y = build_dataset(n=600)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=14, min_samples_leaf=5,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_train)

    print("\n5-fold CV:")
    cv = cross_val_score(clf, X_tr, y_train, cv=5)
    print(f"  {cv.round(3)}  mean={cv.mean():.3f} +/-{cv.std():.3f}")

    y_pred = clf.predict(X_te)
    print("\nTest report:")
    print(classification_report(y_test, y_pred,
          target_names=[CLASSES[i] for i in range(5)]))
    gap = clf.score(X_tr,y_train) - clf.score(X_te,y_test)
    print(f"Overfit gap: {gap*100:.1f}%")

    # Sanity checks
    print("\nSanity checks:")
    sanity = [
        ("B&W bone X-ray",    [0.0,0.0,0.0, 0.197,1639.0,255,110.5,81.2,3.197,0.190,
                                0.0,0.0,0.002,74.9,19.2,101.0,1.07,0.261,0.233,5.0,0.634], "bone"),
        ("Blue bone X-ray",   [14.0,0.201,0.0, 0.197,1639.0,255,110.5,81.2,3.197,0.190,
                                0.0,0.0,0.002,74.9,19.2,101.0,1.07,0.261,0.233,5.0,0.634], "bone"),
        ("Brain MRI (single)",[4.0,0.013,0.010, 0.162,2080.9,250,38.1,50.2,2.079,0.529,
                                0.0,0.0,0.019,41.3,16.4,88.2,4.55,0.556,0.020,1.0,0.392], "mri"),
        # PATCH: composite MRI sanity check
        # Typical 4-panel: cb_ratio~1.0, dark_frac~0.55, multiple peaks
        ("Brain MRI (4-panel)",[5.0,0.015,0.008, 0.180,1800.0,240,42.0,48.0,2.8,0.22,
                                 0.0,0.0,0.018,38.0,14.0,75.0,1.05,0.55,0.015,4.0,0.375], "mri"),
        ("Blue CT scan",      [114.0,0.544,0.0, 0.095,450.0,240,95.0,60.0,3.1,0.10,
                                0.0,0.0,0.004,22.0,9.5,18.0,2.8,0.25,0.18,3.0,0.47], "ct"),
        ("Cameraman B&W",     [0.0,0.0,0.0, 0.146,3031.9,243,118.6,60.9,2.79,0.160,
                                0.0,0.0,0.005,52.98,16.89,45.2,1.05,0.02,0.15,3.0,0.48], "non_medical"),
    ]

    cls_names = {v:k for k,v in CLASSES.items()}
    all_ok = True
    for name, feats, expected in sanity:
        f = np.array(feats, dtype=np.float32).reshape(1,-1)
        probs = clf.predict_proba(scaler.transform(f))[0]
        pred  = CLASSES[clf.predict(scaler.transform(f))[0]]
        thr   = MODALITY_ML_THRESHOLDS.get(expected, 0.40)
        expected_cls = cls_names.get(expected, 0)
        expected_prob = probs[expected_cls] if expected != "non_medical" else probs[0]
        ok = pred == expected
        if not ok: all_ok = False
        status = "OK" if ok else "WRONG"
        print(f"  [{status}] {name:25s} -> {pred:12s} prob={expected_prob:.3f} "
              f"({'PASS' if expected_prob>=thr else f'need>={thr}'})")

    print(f"\nSanity: {'ALL PASSED' if all_ok else 'SOME FAILED'}")

    print("\nTop 8 feature importances:")
    for n,i in sorted(zip(FEATURE_NAMES,clf.feature_importances_),key=lambda x:-x[1])[:8]:
        print(f"  {n:20s} {i:.4f}")

    out = "backend/utils/scan_classifier_multiclass.pkl"
    os.makedirs("backend/utils", exist_ok=True)
    pickle.dump({
        "clf": clf, "scaler": scaler, "classes": CLASSES,
        "thresholds": MODALITY_ML_THRESHOLDS,
        "n_features": 21, "feature_names": FEATURE_NAMES,
    }, open(out,"wb"))
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    train()