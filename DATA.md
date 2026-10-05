# Source corpora and licensing of the derived features

The raw audio is not redistributed here. Download each corpus from its official release and extract
per-layer embeddings with `scripts/extract_embeddings.py`. The cached embeddings we host on OSF
(https://osf.io/8vch9/) are mean-pooled, non-invertible per-layer features (768 dimensions per
utterance); they cannot reconstruct the audio. Throughout we use the 4-class intersection {angry,
happy, neutral, sad}; see the paper for the per-corpus filtering and counts.

**The derived features are released per corpus under that corpus's upstream license.** No single
license covers the whole set. One corpus (ESD) does not permit redistribution of derivatives, so its
embeddings are not hosted; regenerate them from your own licensed copy (see below).

| Corpus | Lang | Corpus license | Embeddings hosted? | Feature license / note |
|---|---|---|---|---|
| EmoDB (Berlin Database of Emotional Speech) | de | attribution-only (CC0 per the audEERING mirror) | yes | CC BY 4.0; cite Burkhardt et al. 2005 |
| RAVDESS (speech subset, not song) | en | CC BY-NC-SA 4.0 | yes | **CC BY-NC-SA 4.0** (non-commercial, share-alike) |
| CREMA-D | en | ODbL 1.0 + DbCL 1.0 | yes | a "Produced Work" under ODbL: attribution notice, no share-alike |
| ESD (Emotional Speech Database, Mandarin half) | zh | signed research-only EULA | **no** | redistribution of derivatives needs NUS's written permission; regenerate via code |
| SUBESCO | bn | CC BY 4.0 | yes | CC BY 4.0; cite Sultana et al. 2021 |
| THAI-SER (external, non-EmoBox test) | th | CC BY-SA 4.0 | yes | **CC BY-SA 4.0** (share-alike) |

## ESD (Mandarin): not redistributed
ESD requires a signed, emailed license before download and restricts redistribution of derivatives,
so we do not host its embeddings. To reproduce the Mandarin results, obtain ESD from
https://github.com/HLTSingapore/Emotional-Speech-Data (complete and email their license), then run

    python scripts/extract_embeddings.py --model <hf-id> --out $DATA_CACHE

on the Mandarin subset to regenerate the `esd_zh__*` arrays locally. Every other corpus's embeddings
are on OSF, so only this one corpus needs local regeneration.

## Attribution
Cite each corpus you use, and note that the hosted features are *modified* (extracted embeddings, not
the original audio):

- **EmoDB**: Burkhardt, Paeschke, Rolfes, Sendlmeier, Weiss (2005), *Interspeech*.
- **RAVDESS**: Livingstone and Russo (2018), *PLoS ONE* 13(5):e0196391. CC BY-NC-SA 4.0.
- **CREMA-D**: Cao, Cooper, Keutmann, Gur, Nenkova, Verma (2014), *IEEE Transactions on Affective
  Computing* 5(4). ODbL 1.0.
- **ESD**: Zhou, Sisman, Liu, Li (2022), *Speech Communication* 137. Research-only EULA (not
  redistributed here).
- **SUBESCO**: Sultana, Rahman, Selim, Iqbal (2021), *PLoS ONE* 16(4):e0250173. CC BY 4.0.
- **THAI-SER**: VISTEC-depa AIResearch Institute of Thailand (2021). CC BY-SA 4.0.
