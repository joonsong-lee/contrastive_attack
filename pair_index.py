import os
import random
from collections import defaultdict


def build_lfw_multi_pair_index(labels, n_pairs=5, seed=0):
    """LFW analog of build_multi_pair_index. Takes the integer-label list of an
    indexable LFW dataset (e.g. HuggingFace logasja/lfw) and groups indices by
    identity. Drops identities with fewer than n_pairs+1 samples.
    Returns {src_idx: [partner_idx_1, ..., partner_idx_n_pairs]}."""
    by_identity = defaultdict(list)
    for i, lbl in enumerate(labels):
        by_identity[int(lbl)].append(i)
    rng = random.Random(seed)
    pair_map = {}
    for indices in by_identity.values():
        if len(indices) < n_pairs + 1:
            continue
        for i in indices:
            others = [j for j in indices if j != i]
            partners = [rng.choice(others) for _ in range(n_pairs)]
            pair_map[i] = partners
    return pair_map


def build_pair_index(image_list_path, identity_file_path, seed=0):
    if not os.path.isfile(image_list_path):
        raise FileNotFoundError(
            f"CelebA-HQ image_list.txt not found at {image_list_path}. "
            "Download it from the original CelebA-HQ release "
            "(https://github.com/tkarras/progressive_growing_of_gans)."
        )
    if not os.path.isfile(identity_file_path):
        raise FileNotFoundError(
            f"identity_CelebA.txt not found at {identity_file_path}. "
            "Download it from the CelebA dataset release "
            "(https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)."
        )

    orig_to_identity = {}
    with open(identity_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            orig_to_identity[parts[0]] = int(parts[1])

    hq_to_identity = {}
    with open(image_list_path, "r") as f:
        header = f.readline().strip().split()
        try:
            idx_col = header.index("idx")
            orig_file_col = header.index("orig_file")
        except ValueError:
            idx_col, orig_file_col = 0, 2
        for line in f:
            parts = line.strip().split()
            if len(parts) <= max(idx_col, orig_file_col):
                continue
            hq_idx = int(parts[idx_col])
            orig_file = parts[orig_file_col]
            ident = orig_to_identity.get(orig_file)
            if ident is not None:
                hq_to_identity[hq_idx] = ident

    by_identity = defaultdict(list)
    for hq_idx, ident in hq_to_identity.items():
        by_identity[ident].append(hq_idx)

    rng = random.Random(seed)
    pair_map = {}
    for hq_indices in by_identity.values():
        if len(hq_indices) < 2:
            continue
        for i in hq_indices:
            others = [j for j in hq_indices if j != i]
            pair_map[i] = rng.choice(others)
    return pair_map


def build_multi_pair_index(image_list_path, identity_file_path, n_pairs=4, seed=0):
    """Like build_pair_index but returns {hq_idx: [p1, p2, ..., p_n_pairs]}.
    Samples with replacement if the identity has fewer than n_pairs other images."""
    single = build_pair_index.__code__  # reuse the file-parsing logic
    # --- re-parse (shares the same file-reading code) ---
    if not os.path.isfile(image_list_path):
        raise FileNotFoundError(f"CelebA-HQ image_list.txt not found at {image_list_path}.")
    if not os.path.isfile(identity_file_path):
        raise FileNotFoundError(f"identity_CelebA.txt not found at {identity_file_path}.")

    orig_to_identity = {}
    with open(identity_file_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            orig_to_identity[parts[0]] = int(parts[1])

    hq_to_identity = {}
    with open(image_list_path, "r") as f:
        header = f.readline().strip().split()
        try:
            idx_col = header.index("idx")
            orig_file_col = header.index("orig_file")
        except ValueError:
            idx_col, orig_file_col = 0, 2
        for line in f:
            parts = line.strip().split()
            if len(parts) <= max(idx_col, orig_file_col):
                continue
            hq_idx = int(parts[idx_col])
            orig_file = parts[orig_file_col]
            ident = orig_to_identity.get(orig_file)
            if ident is not None:
                hq_to_identity[hq_idx] = ident

    by_identity = defaultdict(list)
    for hq_idx, ident in hq_to_identity.items():
        by_identity[ident].append(hq_idx)

    rng = random.Random(seed)
    pair_map = {}
    for hq_indices in by_identity.values():
        if len(hq_indices) < 2:
            continue
        for i in hq_indices:
            others = [j for j in hq_indices if j != i]
            partners = [rng.choice(others) for _ in range(n_pairs)]
            pair_map[i] = partners
    return pair_map
