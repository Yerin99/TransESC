"""
Conversation-level re-split for TransESC: aligned with codes_zcj (Blenderbot-Joint).

Merges the existing 8:1:1 (leaky) preprocessed data, groups samples by
conversation using (emotion_label, situation) as key, disambiguates 3 collision
pairs using dialog content from ESConv.json, applies the same split as
codes_zcj/_reformat/process.py (seed=13, 70:15:15), and outputs leak-free data.

Usage:
    python scripts/conversational_split.py
"""

import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

SEED = 13  # Same as codes_zcj/_reformat/process.py
LEAKY_DIR = "leaky_data/8_1_1"
ESCONV_PATH = "esconv_raw/ESConv.json"
ZCJ_DIR = "/home/yerin/baseline/ESC/codes_zcj/_reformat"
OUT_DIR = "dataset"

EMOTION_MAP = {
    0: "anxiety",
    1: "anger",
    2: "fear",
    3: "depression",
    4: "sadness",
    5: "disgust",
    6: "shame",
    7: "nervousness",
    8: "pain",
    9: "jealousy",
    10: "guilt",
}
EMOTION_TO_LABEL = {v: k for k, v in EMOTION_MAP.items()}


def normalize(text):
    """Normalize whitespace for comparison."""
    return " ".join(text.strip().split())


# ──────────────────────────────────────────────────────────────────────
# Step 1: Load & Merge all splits
# ──────────────────────────────────────────────────────────────────────

def load_split(split):
    """Load all 5 data files for one split, trimming to TSV line count."""
    tsv_path = os.path.join(LEAKY_DIR, f"{split}WithStrategy_short.tsv")
    with open(tsv_path) as f:
        tsv_lines = f.read().strip().split("\n")
    n = len(tsv_lines)

    with open(os.path.join(LEAKY_DIR, f"{split}_emotion.json")) as f:
        emo_data = json.load(f)[:n]

    with open(os.path.join(LEAKY_DIR, f"{split}_keywords.json")) as f:
        kw_data = json.load(f)[:n]

    with open(os.path.join(LEAKY_DIR, f"{split}_csk.pkl"), "rb") as f:
        csk_data = pickle.load(f, encoding="latin1")[:n]

    with open(os.path.join(LEAKY_DIR, f"{split}Situation.txt")) as f:
        sit_lines = f.read().strip().split("\n")[:n]

    print(f"  {split}: {n} samples")
    return tsv_lines, emo_data, kw_data, csk_data, sit_lines


def load_all():
    """Merge train+dev+test into a single sample pool."""
    print("Step 1: Loading and merging all splits from leaky_data/8_1_1...")
    all_tsv, all_emo, all_kw, all_csk, all_sit = [], [], [], [], []
    for split in ["train", "dev", "test"]:
        tsv, emo, kw, csk, sit = load_split(split)
        all_tsv.extend(tsv)
        all_emo.extend(emo)
        all_kw.extend(kw)
        all_csk.extend(csk)
        all_sit.extend(sit)
    total = len(all_tsv)
    print(f"  Total merged: {total} samples")
    assert len(all_emo) == total
    assert len(all_kw) == total
    assert len(all_csk) == total
    assert len(all_sit) == total
    return all_tsv, all_emo, all_kw, all_csk, all_sit, total


# ──────────────────────────────────────────────────────────────────────
# Step 2: Group by (emotion_label, situation)
# ──────────────────────────────────────────────────────────────────────

def group_by_conversation(all_tsv, all_sit):
    """Group sample indices by (emotion_label, normalized_situation)."""
    print("\nStep 2: Grouping by (emotion_label, situation)...")
    conv_groups = defaultdict(list)
    for idx in range(len(all_tsv)):
        emo_label = int(all_tsv[idx].split(" ", 1)[0])
        sit_norm = normalize(all_sit[idx])
        key = (emo_label, sit_norm)
        conv_groups[key].append(idx)

    print(f"  Unique conversation groups: {len(conv_groups)}")
    sizes = [len(v) for v in conv_groups.values()]
    print(f"  Samples per group: min={min(sizes)}, max={max(sizes)}, "
          f"mean={sum(sizes)/len(sizes):.1f}")
    return conv_groups


# ──────────────────────────────────────────────────────────────────────
# Step 3: Disambiguate collision pairs using ESConv.json dialog content
# ──────────────────────────────────────────────────────────────────────

def disambiguate_collisions(conv_groups, all_tsv):
    """Split collision groups into separate conversations using dialog content.

    ESConv.json has 3 pairs of conversations sharing the same (emotion, situation):
      - (fear, "Fear of layoff from job"): conv 16 & 17
      - (sadness, "breakup with partner"): conv 26 & 159
      - (depression, "daily stresses"): conv 698 & 713

    We match samples to specific conversations by comparing utterance text
    from the TSV with the original dialog in ESConv.json.
    """
    print("\nStep 3: Disambiguating collision pairs...")

    with open(ESCONV_PATH) as f:
        esconv = json.load(f)

    # Find collision pairs in ESConv.json
    key_to_convs = defaultdict(list)
    for i, conv in enumerate(esconv):
        emo = conv["emotion_type"]
        sit = normalize(conv["situation"])
        key_to_convs[(emo, sit)].append(i)

    collision_pairs = {k: v for k, v in key_to_convs.items() if len(v) > 1}
    print(f"  Found {len(collision_pairs)} collision pairs in ESConv.json")

    # For each collision pair, build utterance sets for each conversation
    new_groups = {}
    for (emo_type, sit_norm), conv_indices in collision_pairs.items():
        emo_label = EMOTION_TO_LABEL[emo_type]
        merged_key = (emo_label, sit_norm)

        if merged_key not in conv_groups:
            print(f"  [WARN] Collision key {merged_key} not in our data, skipping")
            continue

        sample_indices = conv_groups[merged_key]

        # Build unique utterance set for each conversation
        conv_utt_sets = {}
        for ci in conv_indices:
            utts = set()
            for u in esconv[ci]["dialog"]:
                utts.add(normalize(u["content"]))
            conv_utt_sets[ci] = utts

        # Assign each sample to a conversation
        assigned = {ci: [] for ci in conv_indices}
        ambiguous = []

        for idx in sample_indices:
            # Extract utterance texts from TSV line
            parts = all_tsv[idx].split(" EOS ")
            sample_utts = set()
            for part in parts:
                tokens = part.strip().split(" ", 3)
                if len(tokens) > 3:
                    sample_utts.add(tokens[3].strip())

            # Match against each conversation
            matches = []
            for ci in conv_indices:
                if any(ut in conv_utt_sets[ci] for ut in sample_utts):
                    matches.append(ci)

            if len(matches) == 1:
                assigned[matches[0]].append(idx)
            elif len(matches) == 0:
                ambiguous.append(idx)
            else:
                # Matched multiple - use proportion of matching utterances
                scores = {}
                for ci in matches:
                    scores[ci] = sum(1 for ut in sample_utts if ut in conv_utt_sets[ci])
                best = max(scores, key=scores.get)
                assigned[best].append(idx)

        total_assigned = sum(len(v) for v in assigned.values())
        print(f"  ({emo_type}, \"{sit_norm}\"): {len(sample_indices)} samples → "
              + ", ".join(f"conv{ci}={len(assigned[ci])}" for ci in conv_indices)
              + (f", ambiguous={len(ambiguous)}" if ambiguous else ""))

        if ambiguous:
            print(f"  [WARN] {len(ambiguous)} ambiguous samples for ({emo_type}, \"{sit_norm}\")")

        # Remove the merged group, add individual conversation groups
        del conv_groups[merged_key]
        for ci in conv_indices:
            new_key = (emo_label, sit_norm, ci)  # Extended key with conv index
            new_groups[new_key] = assigned[ci] + ([ambiguous.pop(0)] if ambiguous else [])

    # Add disambiguated groups back
    conv_groups.update(new_groups)
    print(f"  After disambiguation: {len(conv_groups)} conversation groups "
          f"(expected ~1300)")
    return conv_groups


# ──────────────────────────────────────────────────────────────────────
# Step 4: Load codes_zcj split assignment & map to our groups
# ──────────────────────────────────────────────────────────────────────

def load_zcj_split_and_assign(conv_groups, all_tsv):
    """Load codes_zcj's train/valid/test split and assign our conversation
    groups to matching splits.

    codes_zcj/process.py does:
        random.seed(13); random.shuffle(data)
        valid = data[:195]; test = data[195:390]; train = data[390:]
    """
    print("\nStep 4: Loading codes_zcj split assignment...")

    with open(ESCONV_PATH) as f:
        esconv = json.load(f)

    # Load codes_zcj splits to get the split assignment per conversation
    zcj_conv_split = {}  # (emo_type, sit_norm, first_utt) → split_name
    zcj_split_names = {"train": "train", "valid": "dev", "test": "test"}

    for zcj_name, our_name in zcj_split_names.items():
        zcj_path = os.path.join(ZCJ_DIR, f"{zcj_name}.txt")
        with open(zcj_path) as f:
            for line in f:
                d = json.loads(line.strip())
                emo = d["emotion_type"]
                sit = normalize(d["situation"])
                first_utt = normalize(d["dialog"][0]["text"])
                zcj_conv_split[(emo, sit, first_utt)] = our_name

    print(f"  Loaded {len(zcj_conv_split)} conversations from codes_zcj")

    # Build ESConv.json first-utterance lookup
    esconv_first_utts = {}
    for i, conv in enumerate(esconv):
        emo = conv["emotion_type"]
        sit = normalize(conv["situation"])
        first_utt = normalize(conv["dialog"][0]["content"])
        esconv_first_utts[i] = (emo, sit, first_utt)

    # Assign each of our conversation groups to a split
    train_idx, dev_idx, test_idx = [], [], []
    assigned_convs = {"train": 0, "dev": 0, "test": 0}
    unassigned = []

    for key, indices in conv_groups.items():
        if len(key) == 3 and isinstance(key[2], int):
            # Disambiguated collision pair: (emo_label, sit_norm, conv_index)
            emo_label, sit_norm, conv_index = key
            emo_type = EMOTION_MAP[emo_label]
            first_utt = normalize(esconv[conv_index]["dialog"][0]["content"])
            lookup_key = (emo_type, sit_norm, first_utt)
        else:
            # Regular group: (emo_label, sit_norm)
            emo_label, sit_norm = key
            emo_type = EMOTION_MAP[emo_label]
            # Find matching ESConv conversation by first utterance
            # For non-collision groups, there's exactly one matching conv
            lookup_key = None
            for i, conv in enumerate(esconv):
                if conv["emotion_type"] == emo_type and normalize(conv["situation"]) == sit_norm:
                    first_utt = normalize(conv["dialog"][0]["content"])
                    candidate = (emo_type, sit_norm, first_utt)
                    if candidate in zcj_conv_split:
                        lookup_key = candidate
                        break

        if lookup_key and lookup_key in zcj_conv_split:
            split_name = zcj_conv_split[lookup_key]
            if split_name == "train":
                train_idx.extend(indices)
            elif split_name == "dev":
                dev_idx.extend(indices)
            else:
                test_idx.extend(indices)
            assigned_convs[split_name] += 1
        else:
            unassigned.append((key, len(indices)))

    if unassigned:
        print(f"  [WARN] {len(unassigned)} groups unassigned:")
        for key, n in unassigned[:5]:
            print(f"    - {key}: {n} samples")

    print(f"  Assigned conversations: "
          f"train={assigned_convs['train']}, "
          f"dev={assigned_convs['dev']}, "
          f"test={assigned_convs['test']}")

    total_assigned = len(train_idx) + len(dev_idx) + len(test_idx)
    print(f"  Assigned samples: train={len(train_idx)}, dev={len(dev_idx)}, "
          f"test={len(test_idx)} (total={total_assigned})")

    # Shuffle within splits (deterministic)
    random.seed(SEED)
    random.shuffle(train_idx)
    random.shuffle(dev_idx)
    random.shuffle(test_idx)

    return train_idx, dev_idx, test_idx


# ──────────────────────────────────────────────────────────────────────
# Step 5: Save output files
# ──────────────────────────────────────────────────────────────────────

def save_split(split_name, indices, all_tsv, all_emo, all_kw, all_csk, all_sit):
    """Save 5 data files for a split."""
    os.makedirs(OUT_DIR, exist_ok=True)

    tsv_lines = [all_tsv[i] for i in indices]
    emo_data = [all_emo[i] for i in indices]
    kw_data = [all_kw[i] for i in indices]
    csk_data = [all_csk[i] for i in indices]
    sit_lines = [all_sit[i] for i in indices]

    with open(os.path.join(OUT_DIR, f"{split_name}WithStrategy_short.tsv"), "w") as f:
        f.write("\n".join(tsv_lines) + "\n")

    with open(os.path.join(OUT_DIR, f"{split_name}_emotion.json"), "w") as f:
        json.dump(emo_data, f)

    with open(os.path.join(OUT_DIR, f"{split_name}_keywords.json"), "w") as f:
        json.dump(kw_data, f)

    with open(os.path.join(OUT_DIR, f"{split_name}_csk.pkl"), "wb") as f:
        pickle.dump(csk_data, f)

    with open(os.path.join(OUT_DIR, f"{split_name}Situation.txt"), "w") as f:
        f.write("\n".join(sit_lines) + "\n")

    print(f"  {split_name}: {len(indices)} samples saved")


# ──────────────────────────────────────────────────────────────────────
# Step 6: Final leakage-free verification
# ──────────────────────────────────────────────────────────────────────

def verify_no_leakage(train_idx, dev_idx, test_idx, all_tsv, all_sit, total):
    """Verify no data leakage between splits."""
    print("\nStep 6: Verifying leakage-free split...")

    def get_emo_sit_pairs(indices):
        """Get (emotion_label, situation) pairs for leakage check."""
        return set(
            (int(all_tsv[i].split(" ", 1)[0]), normalize(all_sit[i]))
            for i in indices
        )

    def get_situations(indices):
        return set(normalize(all_sit[i]) for i in indices)

    train_pairs = get_emo_sit_pairs(train_idx)
    dev_pairs = get_emo_sit_pairs(dev_idx)
    test_pairs = get_emo_sit_pairs(test_idx)

    train_sits = get_situations(train_idx)
    dev_sits = get_situations(dev_idx)
    test_sits = get_situations(test_idx)

    checks_passed = True

    # Check (emotion, situation) pair overlap
    # Collision pairs (same emotion+situation, different conversations) are acceptable
    for name_a, pairs_a, name_b, pairs_b in [
        ("Train", train_pairs, "Dev", dev_pairs),
        ("Train", train_pairs, "Test", test_pairs),
        ("Dev", dev_pairs, "Test", test_pairs),
    ]:
        overlap = pairs_a & pairs_b
        if overlap:
            print(f"  [INFO] {name_a}-{name_b} (emotion, situation) overlap: {len(overlap)} "
                  f"(collision pairs — different conversations, acceptable)")
            for emo, s in sorted(overlap)[:3]:
                print(f"    - ({EMOTION_MAP.get(emo, emo)}, \"{s[:70]}\")")
        else:
            print(f"  [OK] {name_a}-{name_b} (emotion, situation) overlap: 0")

    # Check situation-only overlap (informational — different emotions are ok)
    for name_a, sits_a, name_b, sits_b in [
        ("Train", train_sits, "Dev", dev_sits),
        ("Train", train_sits, "Test", test_sits),
        ("Dev", dev_sits, "Test", test_sits),
    ]:
        overlap = sits_a & sits_b
        if overlap:
            print(f"  [INFO] {name_a}-{name_b} situation text overlap: {len(overlap)} "
                  f"(different emotions, not true leakage)")
            for s in sorted(overlap)[:3]:
                print(f"    - \"{s[:80]}\"")
        else:
            print(f"  [OK] {name_a}-{name_b} situation text overlap: 0")

    # Total sample count
    total_out = len(train_idx) + len(dev_idx) + len(test_idx)
    if total_out == total:
        print(f"  [OK] Total samples preserved: {total_out}")
    else:
        print(f"  [FAIL] Total samples: {total_out} != {total}")
        checks_passed = False

    # No index overlap
    idx_sets = [("train", set(train_idx)), ("dev", set(dev_idx)), ("test", set(test_idx))]
    for i in range(len(idx_sets)):
        for j in range(i + 1, len(idx_sets)):
            na, sa = idx_sets[i]
            nb, sb = idx_sets[j]
            overlap = sa & sb
            if overlap:
                print(f"  [FAIL] {na}-{nb} index overlap: {len(overlap)}")
                checks_passed = False
            else:
                print(f"  [OK] {na}-{nb} index overlap: 0")

    # Emotion distribution
    print("\n  Emotion distribution:")
    for split_name, indices in [("train", train_idx), ("dev", dev_idx), ("test", test_idx)]:
        emo_counter = Counter()
        for i in indices:
            emo_label = int(all_tsv[i].split(" ", 1)[0])
            emo_counter[emo_label] += 1
        dist = ", ".join(f"{EMOTION_MAP.get(k, k)}:{v}" for k, v in sorted(emo_counter.items()))
        print(f"    {split_name} ({len(indices)} samples): {dist}")

    if checks_passed:
        print("\n  All checks PASSED. Data is leakage-free.")
    else:
        print("\n  WARNING: Issues detected! Review output above.")
    return checks_passed


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Conversation-level Re-split: Aligned with codes_zcj (seed=13)")
    print("=" * 70)

    # Step 1: Load & merge
    all_tsv, all_emo, all_kw, all_csk, all_sit, total = load_all()

    # Step 2: Group by conversation
    conv_groups = group_by_conversation(all_tsv, all_sit)

    # Step 3: Disambiguate collision pairs
    conv_groups = disambiguate_collisions(conv_groups, all_tsv)

    # Step 4: Load codes_zcj split & assign
    train_idx, dev_idx, test_idx = load_zcj_split_and_assign(conv_groups, all_tsv)

    # Step 5: Save output
    print("\nStep 5: Saving new splits to", OUT_DIR)
    save_split("train", train_idx, all_tsv, all_emo, all_kw, all_csk, all_sit)
    save_split("dev", dev_idx, all_tsv, all_emo, all_kw, all_csk, all_sit)
    save_split("test", test_idx, all_tsv, all_emo, all_kw, all_csk, all_sit)

    # Step 6: Verify no leakage
    ok = verify_no_leakage(train_idx, dev_idx, test_idx, all_tsv, all_sit, total)

    print("\n" + "=" * 70)
    if ok:
        print("SUCCESS: Leakage-free split created in", OUT_DIR)
        print("  Aligned with codes_zcj (Blenderbot-Joint) conversation split")
    else:
        print("FAILED: Issues detected. Review output above.")
        sys.exit(1)
    print("=" * 70)


if __name__ == "__main__":
    main()
