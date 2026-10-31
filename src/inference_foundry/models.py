import hashlib
import platform
from functools import lru_cache
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pooled(images):
    array = np.asarray(images, dtype=float)
    return array.reshape(-1, 4, 2, 4, 2).mean(axis=(2, 4)).reshape(-1, 16)


@lru_cache(maxsize=1)
def train_bundle():
    digits = load_digits()
    indexes = np.arange(len(digits.target))
    train, heldout = train_test_split(indexes, test_size=.4, random_state=17, stratify=digits.target)
    calibration, test = train_test_split(heldout, test_size=.5, random_state=19, stratify=digits.target[heldout])
    cheap = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000, random_state=17))
    strong = ExtraTreesClassifier(n_estimators=80, random_state=17, n_jobs=1)
    cheap.fit(pooled(digits.data[train]), digits.target[train])
    strong.fit(digits.data[train], digits.target[train])
    for index in train[:10]:
        cheap.predict_proba(pooled(digits.data[index:index+1]))
        strong.predict(digits.data[index:index+1])
    digest = hashlib.sha256(cheap[-1].coef_.tobytes() + cheap[-1].intercept_.tobytes() + cheap[0].mean_.tobytes() + cheap[0].scale_.tobytes())
    for tree in strong.estimators_:
        digest.update(tree.tree_.threshold.tobytes())
        digest.update(tree.tree_.feature.tobytes())
        digest.update(tree.tree_.value.tobytes())
    provenance = {"dataset": "scikit-learn bundled digits", "split_seeds": [17, 19], "train_count": len(train), "calibration_count": len(calibration), "test_count": len(test), "split_sha256": {name: hashlib.sha256(digits.data[ids].tobytes() + digits.target[ids].tobytes() + ids.tobytes()).hexdigest() for name, ids in [("train", train), ("calibration", calibration), ("test", test)]}, "model_bundle_sha256": digest.hexdigest(), "cheap_model": "2x2 average pooling + StandardScaler + LogisticRegression on 16 features", "strong_model": "ExtraTreesClassifier, 80 trees, 64 input features", "python": platform.python_version(), "sklearn": sklearn.__version__, "platform": platform.system()}
    return {"cheap": cheap, "strong": strong, "data": digits.data, "labels": digits.target, "train": train, "calibration": calibration, "test": test, "provenance": provenance}
