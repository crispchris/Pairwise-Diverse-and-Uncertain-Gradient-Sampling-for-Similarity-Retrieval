import os, torch, pandas as pd, numpy as np, random, json

DATA_DIR = "./ngs_highlights/play_data"
TARGET_NUM_ENTITIES, TARGET_NUM_MOMENTS = 23, 50
SEED = 42

def normalize_tsv(path, win=TARGET_NUM_MOMENTS, stride=50):
    """Convert one .tsv play into sliding-window events of shape (2,23,win)."""
    df = pd.read_csv(path, sep="\t")

    if not (("frame" in df.columns or "frameId" in df.columns) and
            ("x" in df.columns and "y" in df.columns)):
        # not a tracking file, skip
        return []

    # --- entity identifiers ---
    if "displayName" in df.columns:
        entities = ['ball'] + [n for n in df['displayName'].unique() if n != 'ball']
        get_mask = lambda nm: (df['displayName']==nm)
    elif "nflId" in df.columns:
        entities = ['ball'] + [str(n) for n in df['nflId'].dropna().unique()]
        get_mask = lambda nm: (df['nflId'].astype(str)==nm)
    else:
        raise ValueError(f"No usable entity column in {path}")

    eventos = []
    trajs = {}

    for nm in entities:
        if nm == "ball":
            mask = (df.get("displayName")=="ball") if "displayName" in df.columns else (df.get("team")=="football")
        else:
            mask = get_mask(nm)
        ent = df[mask]
        # order by frame (some files call it frame, some frameId)
        sortcol = "frame" if "frame" in ent.columns else "frameId"
        ent = ent.sort_values(sortcol)
        trajs[nm] = (ent["x"].values, ent["y"].values)

    # --- sliding windows with padding ---
    T = max(len(v[0]) for v in trajs.values())
    for start in range(0, T - win + 1, stride):
        xmat = np.zeros((TARGET_NUM_ENTITIES, win))
        ymat = np.zeros((TARGET_NUM_ENTITIES, win))
        for i,nm in enumerate(entities):
            xs,ys = trajs[nm]
            segx,segy = xs[start:start+win], ys[start:start+win]
            xmat[i,:len(segx)] = segx
            ymat[i,:len(segy)] = segy
        eventos.append(np.stack([xmat,ymat]))
    return eventos


def build_tensor_dataset():
    rng = random.Random(SEED)
    fnames = [f for f in sorted(os.listdir(DATA_DIR))
              if f.endswith(".tsv") and "index" not in f.lower()]

    rng.shuffle(fnames)

    n_total = len(fnames)
    n_test, n_val = n_total//10, n_total//10
    test_f, val_f, train_f = fnames[:n_test], fnames[n_test:n_test+n_val], fnames[n_test+n_val:]

    splits = {}
    for name, subset in [("train",train_f),("val",val_f),("test",test_f)]:
        all_events = []
        for f in subset:
            evs = normalize_tsv(os.path.join(DATA_DIR,f))
            if evs: all_events.extend(evs)
        if all_events:
            tensor = torch.tensor(np.stack(all_events), dtype=torch.float32)
            torch.save(tensor, f"nfl_ngs_{name}.pt")
            splits[name] = (tensor.shape, subset)
            print(f"{name}: {tensor.shape}, {len(subset)} plays")
    with open("nfl_ngs_splits.json","w") as f:
        json.dump({k:v for k,(_,v) in splits.items()}, f, indent=2)

if __name__ == "__main__":
    build_tensor_dataset()
