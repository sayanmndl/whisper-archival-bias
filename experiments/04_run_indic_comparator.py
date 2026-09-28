"""
Indic-tuned Whisper comparator on the same FLEURS utterances scored in
experiments/03_run_full_whisper.py. Reads which wav files were transcribed by
the canonical pipeline (corpus_manifest_full.csv) and reuses them so the
A/B is on identical audio.

Models (community Whisper-large-v2 fine-tunes on Indic speech):
  hi: vasista22/whisper-hindi-large-v2
  bn: anuragshas/whisper-large-v2-bn
  ta: vasista22/whisper-tamil-large-v2
Urdu has no large-v2 fine-tune available in this lineage; we omit and flag.

Outputs:
  results/utterance_results_indic.csv
  results/per_language_summary_indic.csv
  results/comparator_provenance.json
"""
import json
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, GenerationConfig

from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "hi": "vasista22/whisper-hindi-large-v2",
    "bn": "anuragshas/whisper-large-v2-bn",
    "ta": "vasista22/whisper-tamil-large-v2",
}
BOOTSTRAP_B = 1000
RNG_SEED = 13
BATCH_SIZE = 4
DEVICE = "cuda"
DTYPE = torch.float16

_english = EnglishTextNormalizer()
_basic = BasicTextNormalizer()


def normalise_ours(s):
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[।,.!?;:\"\(\)\[\]\{\}\-—–_/\\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalise_whisper(s, lang):
    # Per the Whisper paper Appendix D, non-English WER is reported with the
    # BasicTextNormalizer (lowercase, punctuation strip, Unicode NFC), not
    # any language-specific Indic normaliser. We match that protocol so our
    # numbers are apples-to-apples with Radford 2023.
    if s is None:
        return ""
    if lang == "en":
        return _english(s).strip()
    return _basic(s).strip()


def per_utt_wer(ref, hyp):
    if not ref:
        return float("nan")
    return jiwer.wer(ref, hyp)


def bootstrap_ci(values, B=BOOTSTRAP_B, seed=RNG_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    v = np.array([x for x in values if not np.isnan(x)])
    if not len(v):
        return (np.nan, np.nan, np.nan)
    means = np.empty(B)
    for b in range(B):
        means[b] = v[rng.integers(0, len(v), size=len(v))].mean()
    return (float(v.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


# Reuse the wavs the canonical Whisper-large-v3 pipeline wrote.
# FLEURS test has multiple speakers per sample_id; the canonical pipeline
# overwrites the wav each time, so only the LAST speaker per (cfg, sid)
# is on disk. We dedupe both manifest and canonical by (cfg, sid) keep=last
# so the comparator runs on truly distinct audio for the A/B.
manifest = (
    pd.read_csv(RESULTS_DIR / "corpus_manifest_full.csv")
    .drop_duplicates(subset=["fleurs_config", "sample_id"], keep="last")
    .reset_index(drop=True)
)
canonical = (
    pd.read_csv(RESULTS_DIR / "utterance_results_full.csv")
    .drop_duplicates(subset=["fleurs_config", "sample_id"], keep="last")
    .reset_index(drop=True)
)
manifest_keyed = manifest.set_index(["fleurs_config", "sample_id"])
canonical["local_wav"] = canonical.apply(
    lambda r: manifest_keyed.loc[(r["fleurs_config"], r["sample_id"]), "local_wav"], axis=1
)

# References (raw + normalised) are already in canonical
out_rows = []
for w_lang, repo in MODELS.items():
    fl_cfg = {"hi": "hi_in", "bn": "bn_in", "ta": "ta_in"}[w_lang]
    sub = canonical[canonical["fleurs_config"] == fl_cfg].reset_index(drop=True)
    if len(sub) == 0:
        print(f"[skip] no canonical rows for {fl_cfg}")
        continue
    print(f"\n=== {fl_cfg} via {repo} ({len(sub)} utts) ===", flush=True)
    t_load = time.time()
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        repo, torch_dtype=DTYPE, attn_implementation="sdpa"
    ).to(DEVICE)
    processor = AutoProcessor.from_pretrained(repo)
    # vasista22 / anuragshas fine-tunes ship an older generation_config that
    # lacks lang_to_id / task_to_id maps required by transformers 5.x.
    # Overlay a fresh config from openai/whisper-large-v2 (same architecture).
    model.generation_config = GenerationConfig.from_pretrained("openai/whisper-large-v2")
    model.eval()
    print(f"  loaded in {time.time()-t_load:.1f}s", flush=True)

    audios = []
    for wav_rel in sub["local_wav"].tolist():
        arr, sr = sf.read(ROOT / wav_rel)
        if sr != 16000:
            raise RuntimeError(f"sample rate {sr}; expected 16000")
        audios.append(arr.astype(np.float32))

    hyps = []
    t_dec = time.time()
    for i in range(0, len(audios), BATCH_SIZE):
        batch_audio = audios[i:i + BATCH_SIZE]
        inputs = processor(
            batch_audio, sampling_rate=16000, return_tensors="pt", padding=True
        )
        input_feats = inputs.input_features.to(DEVICE, dtype=DTYPE)
        with torch.inference_mode():
            generated = model.generate(
                input_feats,
                language=w_lang,
                task="transcribe",
                max_new_tokens=256,
                num_beams=5,
            )
        decoded = processor.batch_decode(generated, skip_special_tokens=True)
        hyps.extend([d.strip() for d in decoded])
        if (i // BATCH_SIZE) % 5 == 0:
            print(f"  [{i+len(batch_audio):>4}/{len(audios)}] elapsed={time.time()-t_dec:.1f}s",
                  flush=True)
    decode_time = time.time() - t_dec
    print(f"  finished {len(hyps)} hyps in {decode_time:.1f}s "
          f"({decode_time/len(hyps):.2f}s/utt)", flush=True)

    for idx, row in sub.iterrows():
        ref = row["reference"]
        hyp = hyps[idx]
        r_ours = normalise_ours(ref); h_ours = normalise_ours(hyp)
        r_w = normalise_whisper(ref, w_lang); h_w = normalise_whisper(hyp, w_lang)
        out_rows.append({
            "fleurs_config": fl_cfg, "model": "indic_whisper_large_v2",
            "model_repo": repo, "whisper_lang": w_lang,
            "sample_id": row["sample_id"], "duration_s": row["duration_s"],
            "reference": ref, "hypothesis": hyp,
            "wer_ours": per_utt_wer(r_ours, h_ours),
            "wer_whisper": per_utt_wer(r_w, h_w),
        })

    del model, processor
    torch.cuda.empty_cache()

df_out = pd.DataFrame(out_rows)
df_out.to_csv(RESULTS_DIR / "utterance_results_indic.csv", index=False)
print(f"\n[save] {RESULTS_DIR / 'utterance_results_indic.csv'}  ({len(df_out)} rows)")

# Per-language summary including comparison to canonical Whisper-large-v3
canonical_means = canonical.groupby("fleurs_config")[["wer_whisper", "wer_ours"]].mean()
rows = []
for w_lang, repo in MODELS.items():
    fl_cfg = {"hi": "hi_in", "bn": "bn_in", "ta": "ta_in"}[w_lang]
    sub = df_out[df_out["fleurs_config"] == fl_cfg]
    if not len(sub):
        continue
    mean_o, lo_o, hi_o = bootstrap_ci(sub["wer_ours"].values)
    mean_w, lo_w, hi_w = bootstrap_ci(sub["wer_whisper"].values)
    rows.append({
        "fleurs_config": fl_cfg, "model": "indic_whisper_large_v2", "model_repo": repo,
        "n_utterances": len(sub),
        "mean_wer_ours": round(mean_o, 4),
        "ci95_lo_ours": round(lo_o, 4), "ci95_hi_ours": round(hi_o, 4),
        "mean_wer_whisper": round(mean_w, 4),
        "ci95_lo_whisper": round(lo_w, 4), "ci95_hi_whisper": round(hi_w, 4),
        # Unmatched, all-takes canonical mean. 06_make_figures_full.py replaces
        # this with the matched, Radford-normalised value used in Table III.
        "ref_canonical_wer_whisper": round(float(canonical_means.loc[fl_cfg, "wer_whisper"]), 4),
    })
df_sum = pd.DataFrame(rows)
df_sum.to_csv(RESULTS_DIR / "per_language_summary_indic.csv", index=False)
print(f"[save] {RESULTS_DIR / 'per_language_summary_indic.csv'}")
print(df_sum.to_string(index=False))

prov = {
    "models": MODELS,
    "device": DEVICE,
    "dtype": "float16",
    "batch_size": BATCH_SIZE,
    "num_beams": 5,
    "urdu_note": "No vasista22 large-v2 Urdu fine-tune available; omitted.",
    "run_unix_ts": int(time.time()),
}
(RESULTS_DIR / "comparator_provenance.json").write_text(json.dumps(prov, indent=2))
print(f"[save] {RESULTS_DIR / 'comparator_provenance.json'}")
print("\n[done] Indic comparator complete.")
