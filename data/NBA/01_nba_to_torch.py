import os, json, torch, py7zr, random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

import multiprocessing as mp
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


DATA_DIR = "./NBA-Player-Movements/data/2016.NBA.Raw.SportVU.Game.Logs"
TARGET_NUM_ENTITIES, TARGET_NUM_MOMENTS = 11, 150
SEED = 42

def normalize_event(event):
    moments = event["moments"]
    if len(moments) < TARGET_NUM_MOMENTS:
        return None
    x_norm = np.zeros((TARGET_NUM_ENTITIES, TARGET_NUM_MOMENTS))
    y_norm = np.zeros_like(x_norm)
    for t, moment in enumerate(moments[:TARGET_NUM_MOMENTS]):
        entities = moment[5]
        for e, ent in enumerate(entities[:TARGET_NUM_ENTITIES]):
            x_norm[e, t] = ent[2]
            y_norm[e, t] = ent[3]
    return np.stack([x_norm, y_norm])

def clean_and_normalize(events):
    cleaned, last_id, seen = [], 0, []
    for ev in events:
        eid = int(ev["eventId"])
        if eid != last_id + 1 or not ev["moments"]:
            last_id = eid; continue
        mat = normalize_event(ev)
        if mat is None:
            last_id = eid; continue
        if any(np.allclose(mat, prev, atol=1e-5) for prev in seen):
            last_id = eid; continue
        seen.append(mat); cleaned.append(mat); last_id = eid
    return cleaned

def process_archive(fname, seed=SEED):
    """Worker: process one .7z archive deterministically."""
    path = os.path.join(DATA_DIR, fname)
    tensors, total_events = [], 0
    with py7zr.SevenZipFile(path, mode='r') as z:
        for inner_name, bio in z.readall().items():
            if not inner_name.endswith(".json"): continue
            game_json = json.load(bio)
            events = clean_and_normalize(game_json["events"])
            total_events += len(events)
            if not events: continue
            tensors.append(torch.tensor(events, dtype=torch.float32))
    if tensors:
        tensor = torch.cat(tensors, dim=0)
        print(f"[{fname}] total={total_events}, tensor={tensor.shape}")
        return fname, tensor, (fname, total_events)
    else:
        return fname, None, (fname, 0)

def build_split(fnames, num_workers=6):
    """Build dataset tensor for a subset of games."""
    results, stats = {}, []
    with ProcessPoolExecutor(max_workers=num_workers) as exe:
        futures = {exe.submit(process_archive, f, SEED): f for f in fnames}
        for fut in as_completed(futures):
            fname, tensor, stat = fut.result()
            stats.append(stat)
            if tensor is not None:
                results[fname] = tensor
    tensors = [results[f] for f in sorted(results.keys())]
    dataset = torch.cat(tensors, dim=0) if tensors else None
    return dataset, stats

def build_tensor_datasets(num_workers=6):
    # list and deterministically shuffle filenames
    fnames = [f for f in sorted(os.listdir(DATA_DIR)) if f.endswith(".7z")]
    rng = random.Random(SEED)
    rng.shuffle(fnames)

    n_total = len(fnames)
    n_test = n_total // 10
    n_val = n_total // 10
    test_f, val_f, train_f = fnames[:n_test], fnames[n_test:n_test+n_val], fnames[n_test+n_val:]

    print(f"Splitting {n_total} games: train={len(train_f)}, val={len(val_f)}, test={len(test_f)}")

    splits = {}
    for name, subset in [("train", train_f), ("val", val_f), ("test", test_f)]:
        dataset, stats = build_split(subset, num_workers)
        splits[name] = (dataset, stats, subset)
        print(f"\n=== {name.upper()} SUMMARY ===")
        print(f"Games: {len(stats)}")
        print(f"Events: {sum(t for _,t in stats)}")
        if dataset is not None:
            print(f"Tensor shape: {dataset.shape}")
    return splits

if __name__ == "__main__":
    splits = build_tensor_datasets(num_workers=6)
    split_metadata = {}
    for name, (dataset, _, fnames) in splits.items():
        if dataset is not None:
            torch.save(dataset, f"nba_sportvu_{name}.pt")
            print(f"Saved {name} set: nba_sportvu_{name}.pt")
        split_metadata[name] = fnames

    # Save split filenames
    with open("nba_sportvu_splits.json", "w") as f:
        json.dump(split_metadata, f, indent=2)
    print("Saved split metadata: nba_sportvu_splits.json")
