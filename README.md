# Cross-Lingual SER Audit: code and data

Code and data for the TMLR paper *"Re-Examining emotion2vec's Cross-Lingual Advantage: A
Confound-Controlled Comparison of Self-Supervised Speech Models."*

## What's here
- `analysis/`: SUPERB-style probing, cross-lingual transfer (zero-shot and speaker-disjoint
  few-shot, including transfer to and from Thai), RSA, and the INLP speaker-confound control (with a
  rank-matched random-projection control and leave-one-speaker-out cross-validation on the two small
  corpora), plus the baseline-configuration ablation behind the §4.2 result. `superb_probe.py` is the
  SUPERB-style weighted-sum probe used throughout.
- `audit/run.py`: one command to run the per-language probing audit on any SSL encoder.
- `scripts/`: embedding extraction from raw audio.
- `results/`: the result CSVs behind the paper's tables and the appendix per-seed tables, plus the
  192-configuration baseline sweep (`baseline_gap_curve*.csv`, `baseline_recipe_ablation.csv`).

## Setup
    conda env create -f environment.yml
    conda activate ser-audit

## Data
Per-layer embeddings (about 3.7 GB) for five of the six corpora are hosted on OSF. ESD (Mandarin) is
excluded for license reasons; regenerate it locally (see `DATA.md`).

> **Embeddings (OSF):** https://osf.io/8vch9/

Download them and point the cache there (read by `analysis/load.py`):

    export DATA_CACHE=/path/to/embeddings_cache

To regenerate from audio instead, download the corpora (see `DATA.md`) and run
`scripts/extract_embeddings.py`. We redistribute derived features only where the corpus license
permits; ESD (Mandarin) embeddings are not hosted, so regenerate `esd_zh__*` locally from your own
licensed ESD copy (see `DATA.md`). Raw audio is not redistributed.

## Splits
Splits are not stored as files because they are reproducible. Each speaker-independent split is fixed
by a seed (the paper uses 20260505 to 20260509) applied to the `speaker_id` column of a corpus's
metadata, via `speaker_split(meta, seed)`: 60% of speakers train and 40% test, with all of a speaker's
utterances on one side.

## Reproduce a result
Each script in `analysis/` writes its CSVs to `results/`. The main mappings:

| Paper result | Script |
|---|---|
| Per-dataset tables | `probe_superb_full.py`, `probe_superb_full_cv10.py` |
| Speaker-disjoint few-shot transfer | `fewshot_speaker_disjoint.py` |
| Transfer to and from Thai (appendix) | `fewshot_thai_transfer.py` |
| INLP table | `confound_analysis.py` |
| Rank-matched random-projection control | `confound_random_control.py` |
| Leave-one-speaker-out table | `logo_cv.py` |
| RSA table | `rsa_superb.py`, `rsa_permutation_aggregate.py` |
| Figures | `fewshot_figure.py`, `layer_weights_paperfig.py` |

Run a script to regenerate its CSVs from the cached embeddings.

## Audit your own encoder
The evaluation protocol is reusable with any frozen SSL speech encoder.

1. Extract per-layer embeddings on the five corpora:

       python scripts/extract_embeddings.py --model <hf-id> --out $DATA_CACHE

2. Per-language probing audit (the headline arm):

       python audit/run.py --cache $DATA_CACHE --suffix <your-model-file-suffix>

   Prints speaker-independent 4-class accuracy per corpus, averaged over seeds.

3. Full battery: register your encoder in `analysis/load.py` (`MODEL_ALIAS`) and run
   `analysis/fewshot_speaker_disjoint.py` (speaker-disjoint cross-lingual transfer) and
   `analysis/confound_analysis.py` (INLP speaker-confound control).

## License
Code: MIT (see `LICENSE`). Derived features (the hosted embeddings) are released per corpus under each
corpus's upstream license; see `DATA.md`. Notably, RAVDESS is CC BY-NC-SA 4.0 and THAI-SER is CC BY-SA
4.0, and ESD is not redistributed.
