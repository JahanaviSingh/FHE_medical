"""
train_validator.py - Retrained with real feature statistics
Cameraman: lap_var~3000, local_contrast~53, edge_density~0.15
Real X-ray: lap_var~200-600, local_contrast~15-30, edge_density~0.04-0.09
"""

import numpy as np
import cv2
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


def extract_features(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, (128, 128), interpolation=cv2.INTER_AREA)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    r = img[:,:,2].astype(float)
    g = img[:,:,1].astype(float)
    b = img[:,:,0].astype(float)
    gray = (r*0.299 + g*0.587 + b*0.114).astype(np.uint8)
    chroma = abs(r.mean()-g.mean()) + abs(g.mean()-b.mean()) + abs(r.mean()-b.mean())
    max_c = np.maximum(np.maximum(r,g),b)
    min_c = np.minimum(np.minimum(r,g),b)
    sat = (max_c-min_c)/(max_c+1e-6)
    colour_frac = float(((max_c-min_c)>30).mean())
    edges = cv2.Canny(gray, 40, 120)
    edge_density = float(edges.sum())/(255.0*128*128)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_var = float(lap.var())
    int_range = int(gray.max())-int(gray.min())
    hist, _ = np.histogram(gray.flatten(), bins=32, range=(0,255))
    hist_norm = hist.astype(float)/(hist.sum()+1e-6)
    hist_entropy = float(-np.sum(hist_norm*np.log(hist_norm+1e-10)))
    hist_max_frac = float(hist_norm.max())
    green_dom = float(((g>r+10)&(g>b+10)).mean())
    red_dom = float(((r>g+15)&(r>b+15)).mean())
    gray_f = gray.astype(np.float32)
    dct = cv2.dct(gray_f)
    total_e = (dct**2).sum()+1e-6
    h2,w2 = dct.shape
    hf_energy = float(((dct[h2//4:,w2//4:])**2).sum()/total_e)
    kernel = np.ones((8,8),np.float32)/64
    local_mean = cv2.filter2D(gray.astype(float),-1,kernel)
    local_contrast = float(local_mean.std())
    blurred = cv2.GaussianBlur(gray,(15,15),5)
    blob_diff = float(np.abs(gray.astype(float)-blurred.astype(float)).mean())
    return np.array([chroma, float(sat.mean()), colour_frac, edge_density,
                     lap_var, int_range, float(gray.mean()), float(gray.std()),
                     hist_entropy, hist_max_frac, green_dom, red_dom,
                     hf_energy, local_contrast, blob_diff], dtype=np.float32)


def augment(img, rng):
    img = np.clip(img.astype(float) * rng.uniform(0.6, 1.6), 0, 255).astype(np.uint8)
    mean = img.mean()
    img = np.clip((img.astype(float)-mean)*rng.uniform(0.6,1.8)+mean, 0, 255).astype(np.uint8)
    if rng.random() > 0.5: img = cv2.flip(img, 1)
    angle = rng.uniform(-25, 25)
    M = cv2.getRotationMatrix2D((64,64), angle, 1.0)
    img = cv2.warpAffine(img, M, (128,128))
    if rng.random() > 0.4:
        img = cv2.GaussianBlur(img, (0,0), rng.uniform(0.5, 2.5))
    noise = rng.normal(0, rng.uniform(3, 20), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16)+noise, 0, 255).astype(np.uint8)
    if rng.random() > 0.5:
        m = int(rng.uniform(5, 20))
        img = cv2.resize(img[m:-m,m:-m], (128,128))
    return img


def make_medical_scan(rng, modality=None):
    """
    Realistic medical scans.
    Key properties from real data:
      - lap_var: 150-700 (smooth anatomy)
      - local_contrast: 8-28
      - edge_density: 0.03-0.09
      - chroma: near 0
    """
    size = 128
    img = np.zeros((size,size,3), dtype=np.uint8)
    if modality is None:
        modality = rng.choice(["xray","mri","ct","bone"])

    if modality == "xray":
        bg = int(rng.integers(3, 18))
        img[:,:] = bg
        cx = size//2 + int(rng.integers(-8,8))
        cy = size//2 + int(rng.integers(-5,5))
        # Lung fields — smooth gradients
        for y in range(size):
            for x in range(size):
                dx=(x-cx)/(size*rng.uniform(0.32,0.42))
                dy=(y-cy)/(size*rng.uniform(0.38,0.48))
                d2=dx*dx+dy*dy
                if d2<1:
                    v=int(35+rng.uniform(80,130)*(1-d2)**rng.uniform(0.7,1.2))
                    img[y,x]=[max(0,min(255,v))]*3
        # Ribs — moderate brightness lines
        n_ribs = int(rng.integers(4,8))
        for i in range(n_ribs):
            y_r = int(size*0.18 + i*(size*0.65/n_ribs) + rng.integers(-3,3))
            thick = int(rng.integers(2,5))
            for yy in range(max(0,y_r-thick), min(size,y_r+thick)):
                for x in range(int(size*0.12), int(size*0.88)):
                    cur = int(img[yy,x,0])
                    v = min(255, cur + int(rng.integers(40,90)))
                    img[yy,x] = [v,v,v]
        # Spine
        sx = size//2 + int(rng.integers(-3,3))
        for y in range(int(size*0.1), int(size*0.88)):
            v = min(255, int(img[y,sx,0]) + int(rng.integers(60,110)))
            for x in range(max(0,sx-3), min(size,sx+4)):
                img[y,x] = [v,v,v]
        # CRITICAL: strong blur to keep lap_var low (real xray property)
        sigma = rng.uniform(2.5, 5.0)
        for c in range(3):
            img[:,:,c] = cv2.GaussianBlur(img[:,:,c],(0,0),sigma)

    elif modality == "mri":
        img[:,:] = int(rng.integers(0,8))
        cx = size//2+int(rng.integers(-12,12))
        cy = size//2+int(rng.integers(-10,10))
        for y in range(size):
            for x in range(size):
                dx=(x-cx)/(size*rng.uniform(0.30,0.38))
                dy=(y-cy)/(size*rng.uniform(0.35,0.43))
                d2=dx*dx+dy*dy
                if d2<1:
                    base = 175 if d2<0.25 else 110
                    v=int(base*(1-d2*0.25)+rng.integers(-12,12))
                    img[y,x]=[max(0,min(255,v))]*3
        sigma = rng.uniform(2.0, 4.0)
        for c in range(3):
            img[:,:,c] = cv2.GaussianBlur(img[:,:,c],(0,0),sigma)

    elif modality == "ct":
        img[:,:] = int(rng.integers(5,25))
        cx,cy = size//2,size//2
        for y in range(size):
            for x in range(size):
                dx=(x-cx)/(size*0.42); dy=(y-cy)/(size*0.42)
                d2=dx*dx+dy*dy
                if d2<1:
                    v=int(rng.integers(25,220)*(1-d2*0.15))
                    img[y,x]=[max(0,min(255,v))]*3
        sigma = rng.uniform(2.0, 4.5)
        for c in range(3):
            img[:,:,c] = cv2.GaussianBlur(img[:,:,c],(0,0),sigma)

    elif modality == "bone":
        img[:,:] = int(rng.integers(5,20))
        bx = int(size*rng.uniform(0.3,0.7))
        bw = int(size*rng.uniform(0.07,0.13))
        for y in range(size):
            for x in range(size):
                d = abs(x-bx)
                if d < bw:
                    v=int(190*(1-d/bw)+rng.integers(-8,8))
                    img[y,x]=[max(0,min(255,v))]*3
        sigma = rng.uniform(2.5, 4.5)
        for c in range(3):
            img[:,:,c] = cv2.GaussianBlur(img[:,:,c],(0,0),sigma)

    return augment(img, rng)


def make_nonmedical(rng, img_type=None):
    """
    Non-medical images calibrated to real feature values.
    Key: portraits/cameraman have lap_var>1000, local_contrast>35
    """
    size = 128
    img = np.zeros((size,size,3), dtype=np.uint8)
    types = ["bw_portrait","bw_portrait","bw_photo",  # extra weight on hardest cases
             "colour","screenshot","nature","animal","document"]
    if img_type is None:
        img_type = rng.choice(types)

    if img_type in ("bw_portrait", "bw_photo"):
        # Calibrated to match cameraman/lena features:
        # lap_var~1500-4000, local_contrast~35-60, edge_density~0.12-0.20
        bg = int(rng.integers(15, 70))
        img[:,:] = bg

        # Face/head oval
        cx = size//2 + int(rng.integers(-20,20))
        cy = int(size*rng.uniform(0.35,0.55))
        fw = rng.uniform(0.17,0.27)
        fh = rng.uniform(0.23,0.35)
        for y in range(size):
            for x in range(size):
                dx=(x-cx)/(size*fw); dy=(y-cy)/(size*fh)
                if dx*dx+dy*dy < 1:
                    v=int(rng.integers(140,220))
                    img[y,x]=[v,v,v]

        # Dark hair at top — creates sharp edge = high lap_var
        hair_bottom = int(cy - size*fh*0.75)
        img[:max(0,hair_bottom),:] = int(rng.integers(5,35))

        # Sharp clothing/body at bottom
        body_top = int(cy + size*fh*0.85)
        img[min(size-1,body_top):,:] = int(rng.integers(10,50))

        # HIGH FREQUENCY texture — the key differentiator
        # Real portraits have lap_var > 1000 from hair/skin detail
        for _ in range(int(rng.integers(4,8))):
            noise_amp = int(rng.integers(40, 80))
            noise = rng.integers(-noise_amp, noise_amp, (size,size)).astype(np.int16)
            for c in range(3):
                img[:,:,c] = np.clip(img[:,:,c].astype(np.int16)+noise, 0, 255).astype(np.uint8)

        # Sharp edges from face boundary (mimics hair-face transition)
        gray_tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges_tmp = cv2.Canny(gray_tmp, 20, 60)
        kernel = np.ones((3,3),np.uint8)
        edges_tmp = cv2.dilate(edges_tmp, kernel, iterations=2)
        v_edge = int(rng.integers(5,40))
        for c in range(3):
            img[:,:,c] = np.where(edges_tmp>0, v_edge, img[:,:,c]).astype(np.uint8)

        # Minimal blur to keep edges sharp (lap_var stays high)
        if rng.random() > 0.7:
            img = cv2.GaussianBlur(img,(0,0),rng.uniform(0.3,0.8))

    elif img_type == "colour":
        hue = int(rng.integers(0,180))
        for y in range(size):
            for x in range(size):
                h=int((hue+rng.integers(-25,25))%180)
                s=int(rng.integers(80,255)); v=int(rng.integers(60,255))
                bgr=cv2.cvtColor(np.array([[[h,s,v]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0]
                img[y,x]=bgr
        noise=rng.integers(-40,40,(size,size,3)).astype(np.int16)
        img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)

    elif img_type == "screenshot":
        img[:,:] = int(rng.integers(230,255))
        for _ in range(int(rng.integers(3,9))):
            x1,y1=int(rng.integers(0,size-10)),int(rng.integers(0,size-10))
            x2=min(size-1,x1+int(rng.integers(15,60)))
            y2=min(size-1,y1+int(rng.integers(8,25)))
            color=[int(c) for c in rng.integers(30,200,3)]
            cv2.rectangle(img,(x1,y1),(x2,y2),color,-1)
        for _ in range(int(rng.integers(10,25))):
            y=int(rng.integers(5,size-5))
            x1=int(rng.integers(5,25)); x2=int(rng.integers(size//2,size-5))
            c=int(rng.integers(0,60))
            cv2.line(img,(x1,y),(x2,y),(c,c,c),1)

    elif img_type == "nature":
        for y in range(size):
            for x in range(size):
                gv=int(rng.integers(70,190))
                rv=int(gv*rng.uniform(0.2,0.7))
                bv=int(gv*rng.uniform(0.1,0.5))
                img[y,x]=[bv,gv,rv]
        noise=rng.integers(-50,50,(size,size,3)).astype(np.int16)
        img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)

    elif img_type == "animal":
        hue=int(rng.integers(0,30))
        for y in range(size):
            for x in range(size):
                h=int((hue+rng.integers(-15,15))%180)
                s=int(rng.integers(60,180)); v=int(rng.integers(50,180))
                bgr=cv2.cvtColor(np.array([[[h,s,v]]],dtype=np.uint8),cv2.COLOR_HSV2BGR)[0,0]
                img[y,x]=bgr
        for _ in range(4):
            noise=rng.integers(-50,50,(size,size,3)).astype(np.int16)
            img=np.clip(img.astype(np.int16)+noise,0,255).astype(np.uint8)

    elif img_type == "document":
        img[:,:]=int(rng.integers(235,255))
        for _ in range(int(rng.integers(12,28))):
            y=int(rng.integers(5,size-5))
            x1=int(rng.integers(5,20)); x2=int(rng.integers(size//2,size-5))
            c=int(rng.integers(0,50))
            cv2.line(img,(x1,y),(x2,y),(c,c,c),1)

    return augment(img, rng)


def build_dataset(n=600):
    print("Generating training data...")
    rng = np.random.default_rng(42)
    X, y = [], []

    # Medical — 4 modalities
    mods = ["xray","mri","ct","bone"]
    for i in range(n * len(mods)):
        img = make_medical_scan(rng, mods[i % len(mods)])
        X.append(extract_features(img)); y.append(1)
    print(f"  Medical scans: {sum(y)}")

    # Non-medical — extra weight on bw_portrait (hardest case)
    types = ["bw_portrait","bw_portrait","bw_portrait",  # 3x weight
             "bw_photo","colour","screenshot","nature","animal","document"]
    n_before = len(y)
    for i in range(n * len(types)):
        img = make_nonmedical(rng, types[i % len(types)])
        X.append(extract_features(img)); y.append(0)
    print(f"  Non-medical: {len(y)-n_before}")
    print(f"  Total: {len(y)}")
    return np.array(X), np.array(y)


def train():
    X, y = build_dataset(n=600)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=8,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    print("\n5-fold CV:")
    cv = cross_val_score(clf, X_train_s, y_train, cv=5)
    print(f"  {cv.round(3)}  mean={cv.mean():.3f} ±{cv.std():.3f}")

    y_pred = clf.predict(X_test_s)
    print("\nTest set:")
    print(classification_report(y_test, y_pred,
          target_names=["Non-medical","Medical scan"]))
    gap = clf.score(X_train_s,y_train) - clf.score(X_test_s,y_test)
    print(f"Overfit gap: {gap*100:.1f}%")

    # ── Validate on known real-world values ──────────────────────────
    print("\nReal-world sanity check:")
    # Cameraman: lap_var=3031, local_contrast=52.9, chroma=0
    cam_feats = np.array([[0,0,0, 0.1456,3031.9, 243,118.6,60.9,
                           2.79,0.160, 0,0, 0.0045,52.98,16.89]],
                         dtype=np.float32)
    cam_s = scaler.transform(cam_feats)
    cam_p = clf.predict_proba(cam_s)[0][1]
    print(f"  Cameraman (real)  medical_prob={cam_p:.3f}  "
          f"→ {'FAIL (correct)' if cam_p<0.35 else 'PASS (wrong!)'}")

    # Lena: lap_var=1607, chroma=162, sat=0.52
    lena_feats = np.array([[162.3,0.517,0.998, 0.1758,1607.9, 202,123.5,46.5,
                            3.06,0.079, 0,0.945, 0.0035,37.15,18.80]],
                          dtype=np.float32)
    lena_s = scaler.transform(lena_feats)
    lena_p = clf.predict_proba(lena_s)[0][1]
    print(f"  Lena portrait     medical_prob={lena_p:.3f}  "
          f"→ {'FAIL (correct)' if lena_p<0.35 else 'PASS (wrong!)'}")

    # Synthetic X-ray target: lap_var~300, local_contrast~18, chroma~0
    xray_feats = np.array([[1.2,0.008,0.001, 0.055,320.0, 210,85.0,35.0,
                            2.95,0.095, 0,0, 0.002,18.5,8.2]],
                          dtype=np.float32)
    xray_s = scaler.transform(xray_feats)
    xray_p = clf.predict_proba(xray_s)[0][1]
    print(f"  Typical X-ray     medical_prob={xray_p:.3f}  "
          f"→ {'PASS (correct)' if xray_p>=0.35 else 'FAIL (wrong!)'}")

    os.makedirs("backend/utils", exist_ok=True)
    with open("backend/utils/scan_classifier.pkl","wb") as f:
        pickle.dump({"clf":clf,"scaler":scaler,"features":15}, f)
    print("\nModel saved → backend/utils/scan_classifier.pkl")

    print("\nTop feature importances:")
    names=["chroma","sat","colour_frac","edge_density","lap_var",
           "int_range","mean_int","std_int","hist_entropy","hist_max",
           "green_dom","red_dom","hf_energy","local_contrast","blob_diff"]
    for n,i in sorted(zip(names,clf.feature_importances_),key=lambda x:-x[1])[:8]:
        print(f"  {n:20s} {i:.4f}")


if __name__ == "__main__":
    train()
