"""Unified dataset loader for the 5 corpora used in this project.

Each loader returns a list of dicts with the same schema:

    {
        "dataset": str,            # corpus name
        "audio_path": str,         # absolute path to .wav
        "emotion": str,            # one of EMOTION_4CLASS
        "original_emotion": str,   # corpus-specific label, for traceability
        "speaker_id": str,         # corpus-specific speaker identifier
        "sample_rate": int,        # native SR (we resample to 16k at extract time)
    }

Only the 4-class emotion intersection is kept: angry, happy, neutral, sad.
Utterances with other emotions in the source corpus are filtered out.

ESD-ZH additionally subsamples to 100 utterances per (speaker, emotion) cell,
seeded for reproducibility. See docstring on `load_esd_zh`.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

EMOTION_4CLASS = ["angry", "happy", "neutral", "sad"]

DATA_ROOT = Path("/path/to/data/audio")

ESD_ZH_SUBSAMPLE_PER_CELL = 100  # 10 speakers x 4 emotions x 100 = 4,000
ESD_ZH_SEED = 20260504           # seed for reproducible ESD-ZH subsampling


# ---------------------------------------------------------------------------
# EmoDB (German, Berlin Emotional Speech Database)
# ---------------------------------------------------------------------------
# Filename: SS<sentence><emotion-code><take>.wav
#   Speaker: 2-digit (03, 08, 09, 10, 11, 12, 13, 14, 15, 16)
#   Emotion code: W=anger, L=boredom, E=disgust, A=fear,
#                 F=happiness, T=sadness, N=neutral

EMODB_EMOTION_MAP = {
    "W": "angry",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}


def load_emodb(root: Path | str = DATA_ROOT / "emodb") -> list[dict]:
    root = Path(root)
    wav_dir = root / "wav"
    records = []
    for wav in sorted(wav_dir.glob("*.wav")):
        name = wav.stem
        speaker_id = name[:2]
        emotion_code = name[5]
        if emotion_code not in EMODB_EMOTION_MAP:
            continue
        records.append({
            "dataset": "emodb",
            "audio_path": str(wav),
            "emotion": EMODB_EMOTION_MAP[emotion_code],
            "original_emotion": emotion_code,
            "speaker_id": f"emodb_{speaker_id}",
            "sample_rate": 16000,
        })
    return records


# ---------------------------------------------------------------------------
# RAVDESS (English, Ryerson Audio-Visual Database of Emotional Speech and Song)
# ---------------------------------------------------------------------------
# Audio-only speech subset.
# Filename: <Modality>-<VocalChannel>-<Emotion>-<Intensity>-<Statement>-<Repetition>-<Actor>.wav
# Codes:
#   Emotion: 01=neutral 02=calm 03=happy 04=sad 05=angry 06=fearful 07=disgust 08=surprised
#   Actor:   01..24 (odd=male, even=female)
# We keep only speech (vocal channel 01), 4 emotions: neutral, happy, sad, angry.

RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
}


def load_ravdess(root: Path | str = DATA_ROOT / "ravdess") -> list[dict]:
    root = Path(root)
    records = []
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        modality, vocal, emo, intensity, statement, rep, actor = parts
        if vocal != "01":  # speech only, skip song
            continue
        if emo not in RAVDESS_EMOTION_MAP:
            continue
        records.append({
            "dataset": "ravdess",
            "audio_path": str(wav),
            "emotion": RAVDESS_EMOTION_MAP[emo],
            "original_emotion": emo,
            "speaker_id": f"ravdess_{actor}",
            "sample_rate": 48000,  # native, we resample at extract
        })
    return records


# ---------------------------------------------------------------------------
# CREMA-D (English, Crowd-sourced Emotional Multimodal Actors Dataset)
# ---------------------------------------------------------------------------
# Filename: <ActorID>_<SentenceCode>_<EmotionCode>_<LevelCode>.wav
#   ActorID: 1001..1091
#   EmotionCode: ANG, DIS, FEA, HAP, NEU, SAD
# We keep ANG, HAP, NEU, SAD.

CREMA_D_EMOTION_MAP = {
    "ANG": "angry",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


def load_crema_d(root: Path | str = DATA_ROOT / "crema_d") -> list[dict]:
    root = Path(root)
    # Audio is in <root>/repo/AudioWAV/ after git clone
    candidates = [root / "repo" / "AudioWAV", root / "AudioWAV"]
    audio_dir = next((d for d in candidates if d.is_dir()), None)
    if audio_dir is None:
        raise FileNotFoundError(f"CREMA-D AudioWAV not found under {root}")
    records = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) != 4:
            continue
        actor, sent, emo, level = parts
        if emo not in CREMA_D_EMOTION_MAP:
            continue
        records.append({
            "dataset": "crema_d",
            "audio_path": str(wav),
            "emotion": CREMA_D_EMOTION_MAP[emo],
            "original_emotion": emo,
            "speaker_id": f"crema_d_{actor}",
            "sample_rate": 16000,
        })
    return records


# ---------------------------------------------------------------------------
# SUBESCO (Bangla, SUST Bangla Emotional Speech Corpus)
# ---------------------------------------------------------------------------
# Filename structure to be confirmed once downloaded; per the paper, 7 emotions
# (Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise) across 20 speakers.
# Implementation will be added once the actual layout is inspected.

# SUBESCO filename: <Gender>_<SpeakerNum>_<SpeakerName>_S_<SentenceNum>_<EMOTION>_<TakeNum>.wav
# Example: F_01_OISHI_S_10_ANGRY_1.wav
# Emotions in dataset: ANGRY, DISGUST, FEAR, HAPPY, NEUTRAL, SAD, SURPRISE
# We keep ANGRY, HAPPY, NEUTRAL, SAD.

SUBESCO_EMOTION_MAP = {
    "ANGRY": "angry",
    "HAPPY": "happy",
    "NEUTRAL": "neutral",
    "SAD": "sad",
}


def load_subesco(root: Path | str = DATA_ROOT / "subesco") -> list[dict]:
    root = Path(root)
    # Audio is in <root>/SUBESCO/ after unzip
    candidates = [root / "SUBESCO", root]
    audio_dir = next((d for d in candidates if d.is_dir() and any(d.glob("*.wav"))), None)
    if audio_dir is None:
        raise FileNotFoundError(f"SUBESCO: no wav files found under {root}")
    records = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        # Expected length 7: gender, num, name, S, sent, emo, take
        if len(parts) != 7:
            continue
        gender, spk_num, spk_name, _S, sent, emo, take = parts
        if emo not in SUBESCO_EMOTION_MAP:
            continue
        records.append({
            "dataset": "subesco",
            "audio_path": str(wav),
            "emotion": SUBESCO_EMOTION_MAP[emo],
            "original_emotion": emo,
            "speaker_id": f"subesco_{gender}{spk_num}",  # e.g. F01, M05
            "sample_rate": 44100,  # native; resampled to 16k at extract
        })
    return records


# ---------------------------------------------------------------------------
# ESD (Emotional Speech Dataset) — Mandarin half only (speakers 0001..0010)
# ---------------------------------------------------------------------------
# Layout: <root>/<speaker_id>/<emotion>/<file>.wav
#   Mandarin speakers: 0001..0010
#   English speakers: 0011..0020 (excluded)
#   Emotions (folders): Angry, Happy, Neutral, Sad, Surprise
# We keep Angry/Happy/Neutral/Sad and drop Surprise.
# We then subsample to 100 utterances per (speaker, emotion) cell with a fixed seed
# so that ESD-ZH does not dominate the corpus mix in pooled analyses.

ESD_EMOTION_MAP = {
    "Angry": "angry",
    "Happy": "happy",
    "Neutral": "neutral",
    "Sad": "sad",
}

ESD_ZH_SPEAKERS = {f"{i:04d}" for i in range(1, 11)}  # 0001..0010


def load_esd_zh(root: Path | str = DATA_ROOT / "esd_zh") -> list[dict]:
    root = Path(root)
    # ESD's zip extracts into a top-level folder (often "Emotional Speech Dataset (ESD)")
    # locate the speaker directories
    speaker_dirs = []
    for cand in root.rglob("[0-9][0-9][0-9][0-9]"):
        # Skip macOS resource-fork artifacts that some zips create
        if "__MACOSX" in cand.parts:
            continue
        if cand.is_dir() and cand.name in ESD_ZH_SPEAKERS:
            speaker_dirs.append(cand)
    if not speaker_dirs:
        raise FileNotFoundError(f"ESD-ZH: no Mandarin speaker dirs (0001-0010) found under {root}")

    by_cell: dict[tuple[str, str], list[dict]] = {}
    for sp_dir in speaker_dirs:
        speaker = sp_dir.name
        for emo_dir in sp_dir.iterdir():
            if not emo_dir.is_dir() or emo_dir.name not in ESD_EMOTION_MAP:
                continue
            mapped = ESD_EMOTION_MAP[emo_dir.name]
            for wav in emo_dir.glob("*.wav"):
                rec = {
                    "dataset": "esd_zh",
                    "audio_path": str(wav),
                    "emotion": mapped,
                    "original_emotion": emo_dir.name,
                    "speaker_id": f"esd_zh_{speaker}",
                    "sample_rate": 16000,
                }
                by_cell.setdefault((speaker, mapped), []).append(rec)

    rng = random.Random(ESD_ZH_SEED)
    sampled = []
    for cell, recs in sorted(by_cell.items()):
        recs_sorted = sorted(recs, key=lambda r: r["audio_path"])
        if len(recs_sorted) > ESD_ZH_SUBSAMPLE_PER_CELL:
            picks = rng.sample(recs_sorted, ESD_ZH_SUBSAMPLE_PER_CELL)
        else:
            picks = recs_sorted
        sampled.extend(sorted(picks, key=lambda r: r["audio_path"]))
    return sampled


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

LOADERS: dict[str, Callable[..., list[dict]]] = {
    "emodb": load_emodb,
    "ravdess": load_ravdess,
    "crema_d": load_crema_d,
    "subesco": load_subesco,
    "esd_zh": load_esd_zh,
}


# ---------------------------------------------------------------------------
# Full-emotion-set variants (no 4-class filtering) for Claim 1 verification:
# replicating the emotion2vec paper's Table 4 per-dataset full-class numbers.
# ---------------------------------------------------------------------------

EMODB_EMOTION_MAP_FULL = {
    "W": "angry", "F": "happy", "T": "sad", "N": "neutral",
    "L": "boredom", "E": "disgust", "A": "fear",
}

def load_emodb_full(root: Path | str = DATA_ROOT / "emodb") -> list[dict]:
    root = Path(root)
    wav_dir = root / "wav"
    records = []
    for wav in sorted(wav_dir.glob("*.wav")):
        name = wav.stem
        speaker_id = name[:2]
        emotion_code = name[5]
        if emotion_code not in EMODB_EMOTION_MAP_FULL:
            continue
        records.append({
            "dataset": "emodb_full",
            "audio_path": str(wav),
            "emotion": EMODB_EMOTION_MAP_FULL[emotion_code],
            "original_emotion": emotion_code,
            "speaker_id": f"emodb_{speaker_id}",
            "sample_rate": 16000,
        })
    return records


RAVDESS_EMOTION_MAP_FULL = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}

def load_ravdess_full(root: Path | str = DATA_ROOT / "ravdess") -> list[dict]:
    root = Path(root)
    records = []
    for wav in sorted(root.rglob("*.wav")):
        parts = wav.stem.split("-")
        if len(parts) != 7:
            continue
        modality, vocal, emo, intensity, statement, rep, actor = parts
        if vocal != "01":
            continue
        if emo not in RAVDESS_EMOTION_MAP_FULL:
            continue
        records.append({
            "dataset": "ravdess_full",
            "audio_path": str(wav),
            "emotion": RAVDESS_EMOTION_MAP_FULL[emo],
            "original_emotion": emo,
            "speaker_id": f"ravdess_{actor}",
            "sample_rate": 48000,
        })
    return records


SUBESCO_EMOTION_MAP_FULL = {
    "ANGRY": "angry", "DISGUST": "disgust", "FEAR": "fear",
    "HAPPY": "happy", "NEUTRAL": "neutral", "SAD": "sad", "SURPRISE": "surprise",
}

def load_subesco_full(root: Path | str = DATA_ROOT / "subesco") -> list[dict]:
    root = Path(root)
    candidates = [root / "SUBESCO", root]
    audio_dir = next((d for d in candidates if d.is_dir() and any(d.glob("*.wav"))), None)
    if audio_dir is None:
        raise FileNotFoundError(f"SUBESCO: no wav files found under {root}")
    records = []
    for wav in sorted(audio_dir.glob("*.wav")):
        parts = wav.stem.split("_")
        if len(parts) != 7:
            continue
        gender, spk_num, spk_name, _S, sent, emo, take = parts
        if emo not in SUBESCO_EMOTION_MAP_FULL:
            continue
        records.append({
            "dataset": "subesco_full",
            "audio_path": str(wav),
            "emotion": SUBESCO_EMOTION_MAP_FULL[emo],
            "original_emotion": emo,
            "speaker_id": f"subesco_{gender}{spk_num}",
            "sample_rate": 44100,
        })
    return records


LOADERS["emodb_full"] = load_emodb_full
LOADERS["ravdess_full"] = load_ravdess_full
LOADERS["subesco_full"] = load_subesco_full


# ---------------------------------------------------------------------------
# THAI-SER (Thai Speech Emotion Recognition Corpus)
# Test-bed for Claim 2: language NOT in EmoBox, so emotion2vec_plus_* couldn't
# have been contaminated via fine-tuning on it.
# ---------------------------------------------------------------------------
# Layout:
#   thai_ser/studio001/{clip,con,middle}/<file>.flac
#   thai_ser/emotion_label.json -> {filename: [{assigned_emo, majority_emo, agreement, ...}]}
# Filename: s<session>_<segment>_actor<actor>_<scene>_<utt>.flac
# Emotions in dataset: Neutral, Anger, Happiness, Sadness, Frustration (5)
# We map 4 of them to our 4-class set; drop Frustration.

THAI_EMOTION_MAP = {
    "Angry": "angry",
    "Happy": "happy",
    "Sad": "sad",
    "Neutral": "neutral",
    # discarded: Frustrated, None, other
}

THAI_MIN_AGREEMENT = 0.5  # majority vote requires more than half


def load_thai_ser(root: Path | str = DATA_ROOT / "thai_ser",
                  min_agreement: float = THAI_MIN_AGREEMENT) -> list[dict]:
    import json
    root = Path(root)
    label_path = root / "emotion_label.json"
    with open(label_path) as f:
        labels = json.load(f)
    records = []
    for fname, anns in labels.items():
        if not anns:
            continue
        ann = anns[0]  # use first annotation set (there's usually only one per file)
        emo_raw = ann.get("majority_emo")
        agree = ann.get("agreement", 0.0)
        if emo_raw not in THAI_EMOTION_MAP or agree < min_agreement:
            continue
        # Parse filename to extract actor + studio + segment
        # Format: s<session>_<segment>_actor<actor>_<scene>_<utt>.flac
        parts = fname.replace(".flac", "").split("_")
        if len(parts) < 5 or not parts[0].startswith("s") or not parts[2].startswith("actor"):
            continue
        session = parts[0][1:]              # e.g., "001"
        segment = parts[1]                  # clip / con / middle
        actor   = parts[2][len("actor"):]    # e.g., "001"
        # Locate the file
        candidate = root / f"studio{session}" / segment / fname
        if not candidate.exists():
            continue
        records.append({
            "dataset": "thai_ser",
            "audio_path": str(candidate),
            "emotion": THAI_EMOTION_MAP[emo_raw],
            "original_emotion": emo_raw,
            "speaker_id": f"thai_actor{actor}",
            "sample_rate": 44100,  # native; resampled at extract
        })
    return records


LOADERS["thai_ser"] = load_thai_ser


def load_dataset(name: str) -> list[dict]:
    if name not in LOADERS:
        raise ValueError(f"Unknown dataset {name!r}. Known: {list(LOADERS)}")
    return LOADERS[name]()


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "emodb"
    records = load_dataset(name)
    print(f"Loaded {len(records)} records from {name}")
    from collections import Counter
    print("Emotion counts:", dict(Counter(r["emotion"] for r in records)))
    print("Speaker counts:", dict(Counter(r["speaker_id"] for r in records)))
    print("Sample record:", records[0])
