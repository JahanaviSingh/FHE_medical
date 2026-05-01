"""
Patch to add multiclass model support to validator.py.
Run: python backend/utils/validator_multiclass_patch.py
"""
import os

PATCH = '''
    # Multiclass model support
    MULTICLASS_PATH = os.path.join(os.path.dirname(__file__), "scan_classifier_multiclass.pkl")
    MODALITY_TO_CLASS = {"xray":1, "mri":2, "bone":3, "ct":4}

    def _load_model(self):
        # Try multiclass first, fall back to binary
        for path, is_multi in [(self.MULTICLASS_PATH, True), (MODEL_PATH, False)]:
            if os.path.exists(path):
                try:
                    import pickle
                    d = pickle.load(open(path,"rb"))
                    self.model  = d["clf"]
                    self.scaler = d["scaler"]
                    self.is_multiclass = is_multi
                    self.model_classes  = d.get("classes", {0:"non_medical",1:"medical"})
                    print(f"[Validator] {'Multi-class' if is_multi else 'Binary'} model loaded: {os.path.basename(path)}")
                    return
                except Exception as e:
                    print(f"[Validator] Could not load {path}: {e}")
        print("[Validator] No model found — using physics checks only")

    def _ml_prob(self, image) -> float:
        """Return probability that image matches the declared modality."""
        if self.model is None:
            return 0.5
        try:
            feats   = extract_features(image).reshape(1,-1)
            feats_s = self.scaler.transform(feats)
            probs   = self.model.predict_proba(feats_s)[0]

            if self.is_multiclass:
                # Get probability for the specific declared modality
                # We need the modality from the calling context
                # Stored temporarily in self._current_modality
                mod = getattr(self, "_current_modality", "xray")
                cls_idx = self.MODALITY_TO_CLASS.get(mod, 1)
                # Also check non-medical prob (class 0)
                non_med_prob = probs[0]
                med_prob     = probs[cls_idx]
                # Penalise if non-medical is dominant
                if non_med_prob > 0.60:
                    return float(med_prob * 0.3)  # heavy penalty
                return float(med_prob)
            else:
                return float(probs[1])
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"ML prediction failed: {e}")
            return 0.5
'''
print("Patch content ready")
print("Apply this to validator.py _load_model and _ml_prob methods")
