import numpy as np
import os

fname = 'imageDataNew-10-10-5.npy'
if not os.path.exists(fname):
    print('No preprocessed .npy file found:', fname)
    raise SystemExit(1)

data = np.load(fname, allow_pickle=True)
data = list(data)

X = []
y = []

for item in data:
    try:
        img = np.array(item[0]).astype('float32')
        lbl = item[1]
    except Exception:
        continue
    X.append(img.ravel())
    # normalize label if present, else derive a demo label from intensity
    try:
        if isinstance(lbl, (list, tuple, np.ndarray)) and len(lbl) == 2:
            lab = int(lbl[1] == 1)
            y.append(lab)
            continue
        lab = int(str(lbl))
        y.append(lab)
    except Exception:
        # Derive label heuristically for demo: later we'll threshold by median intensity
        y.append(None)

if len(X) == 0:
    print('No samples to train on')
    raise SystemExit(1)

# If labels are all None or single-class, derive binary labels by intensity median
if all(v is None for v in y) or len(set([v for v in y if v is not None])) == 1:
    means = [float(x.mean()) for x in X]
    med = float(np.median(means))
    y = [1 if float(x.mean()) > med else 0 for x in X]

try:
    from sklearn.linear_model import LogisticRegression
    import pickle
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X, y)
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(clf, f)
    print('Saved trained_model.pkl')
except Exception as e:
    print('Failed to train/save model:', e)
    raise
