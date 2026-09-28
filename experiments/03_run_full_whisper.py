"""
Full-FLEURS-test ASR audit pipeline (Whisper-large-v3 fp16 on GPU).

Reviewer-fatal gap closed vs 01_run_pipeline_gpu.py:
  - N per language: 30 -> full test split (capped at N_MAX_PER_LANG=600 for runtime)
  - Reports two WER variants: our orthographic normalisation and the Radford
    2023 Appendix D protocol (EnglishTextNormalizer + BasicTextNormalizer).
  - Adds an unpaired bootstrap mean-difference test (EN vs each Indic) with
    a multiple-comparison correction across the four contrasts.

Outputs (separate filenames from the N=30 pilot, so both stay on disk):
  results/utterance_results_full.csv
  results/per_language_summary_full.csv
  results/pairwise_significance.csv
  results/run_provenance_full.json
  results/corpus_manifest_full.csv
"""
import io
import json
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from faster_whisper import WhisperModel

from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
AUDIO_DIR = ROOT / "experiments" / "audio_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGES = [
    ("en_us", "en", "English (US)", "en"),
    ("hi_in", "hi", "Hindi",        "south_asia"),
    ("bn_in", "bn", "Bengali",      "south_asia"),
    ("ta_in", "ta", "Tamil",        "south_asia"),
    ("ur_pk", "ur", "Urdu",         "south_asia"),
]
N_MAX_PER_LANG = 600
WHISPER_MODEL_SIZE = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"
BEAM_SIZE = 5
BOOTSTRAP_B = 1000
RNG_SEED = 13
MAX_UTT_SECONDS = 35.0

_basic = BasicTextNormalizer()
_english = EnglishTextNormalizer()


def normalise_ours(s: str) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFC", s).lower()
    s = re.sub(r"[।,.!?;:\"\(\)\[\]\{\}\-—–_/\\]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalise_whisper(s: str, lang: str) -> str:
    # Per Appendix D of Radford et al. 2023: EnglishTextNormalizer for English,
    # BasicTextNormalizer for non-English. No language-specific Indic normalisers.
    if s is None:
        return ""
    if lang == "en":
        return _english(s).strip()
    return _basic(s).strip()


def per_utt_wer(ref_norm: str, hyp_norm: str) -> float:
    if not ref_norm:
        return float("nan")
    return jiwer.wer(ref_norm, hyp_norm)


def bootstrap_ci(values, B=BOOTSTRAP_B, seed=RNG_SEED, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.array([v for v in values if not np.isnan(v)])
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    n = len(values)
    means = np.empty(B)
    for b in range(B):
        means[b] = values[rng.integers(0, n, size=n)].mean()
    return (float(values.mean()),
            float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


def unpaired_bootstrap_diff(values_a, values_b, B=BOOTSTRAP_B, seed=RNG_SEED):
    """Unpaired bootstrap of the mean WER difference (b - a).

    a and b are unpaired (different utterances across two language subsets).
    Returns observed diff and a two-sided bootstrap p-value via the quantile
    method. Reported p-values are floored at 1/B since the quantile estimate
    cannot resolve below that threshold.
    """
    rng = np.random.default_rng(seed)
    a = np.array([v for v in values_a if not np.isnan(v)])
    b = np.array([v for v in values_b if not np.isnan(v)])
    obs = b.mean() - a.mean()
    diffs = np.empty(B)
    for i in range(B):
        diffs[i] = (b[rng.integers(0, len(b), size=len(b))].mean()
                    - a[rng.integers(0, len(a), size=len(a))].mean())
    p_two = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return float(obs), float(p_two), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


print(f"[init] Loading Whisper-{WHISPER_MODEL_SIZE} on {WHISPER_DEVICE} ({WHISPER_COMPUTE})...")
t0 = time.time()
model = WhisperModel(WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
print(f"[init] Loaded in {time.time()-t0:.1f}s")

utt_rows = []
manifest_rows = []

for fl_cfg, w_lang, display, region in LANGUAGES:
    print(f"\n=== {display} ({fl_cfg}) ===", flush=True)
    t_lang0 = time.time()
    ds = load_dataset("google/fleurs", fl_cfg, split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    it = iter(ds)

    n_done = 0
    while n_done < N_MAX_PER_LANG:
        try:
            sample = next(it)
        except StopIteration:
            print(f"  [info] stream exhausted at {n_done}/{N_MAX_PER_LANG}", flush=True)
            break
        ref = sample.get("transcription") or sample.get("raw_transcription") or ""
        if not ref.strip():
            continue
        audio_dict = sample["audio"]
        audio_bytes = audio_dict.get("bytes")
        if not audio_bytes:
            continue
        try:
            arr, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception as e:
            print(f"  [skip] decode error: {e}", flush=True); continue
        if len(arr) / sr > MAX_UTT_SECONDS:
            continue
        wav_path = AUDIO_DIR / f"{fl_cfg}_{sample['id']}.wav"
        sf.write(wav_path, arr, sr)
        t_dec = time.time()
        segments, info = model.transcribe(str(wav_path), beam_size=BEAM_SIZE, language=w_lang)
        hyp = " ".join(seg.text.strip() for seg in segments)
        decode_t = time.time() - t_dec

        ref_ours = normalise_ours(ref)
        hyp_ours = normalise_ours(hyp)
        wer_ours = per_utt_wer(ref_ours, hyp_ours)

        ref_w = normalise_whisper(ref, w_lang)
        hyp_w = normalise_whisper(hyp, w_lang)
        wer_w = per_utt_wer(ref_w, hyp_w)

        utt_rows.append({
            "fleurs_config": fl_cfg, "whisper_lang": w_lang, "display": display, "region": region,
            "sample_id": sample["id"], "duration_s": round(len(arr) / sr, 3), "sampling_rate": sr,
            "reference": ref, "hypothesis": hyp,
            "reference_norm_ours": ref_ours, "hypothesis_norm_ours": hyp_ours, "wer_ours": wer_ours,
            "reference_norm_whisper": ref_w, "hypothesis_norm_whisper": hyp_w, "wer_whisper": wer_w,
            "lid_prob": float(info.language_probability),
            "lid_detected": info.language, "decode_s": round(decode_t, 3),
        })
        manifest_rows.append({
            "source": "google/fleurs", "fleurs_config": fl_cfg,
            "display": display, "region": region, "sample_id": sample["id"],
            "duration_s": round(len(arr) / sr, 3), "sampling_rate": sr,
            "license": "CC-BY 4.0",
            "url": "https://huggingface.co/datasets/google/fleurs",
            "local_wav": str(wav_path.relative_to(ROOT)),
        })

        n_done += 1
        if n_done % 25 == 0 or n_done == 1:
            print(f"  [{n_done:>4}/{N_MAX_PER_LANG}] dur={len(arr)/sr:.1f}s "
                  f"decode={decode_t:.2f}s WER(ours)={wer_ours:.3f} WER(w)={wer_w:.3f}",
                  flush=True)
    print(f"  done {n_done} utterances in {time.time()-t_lang0:.1f}s", flush=True)
    pd.DataFrame(utt_rows).to_csv(RESULTS_DIR / "utterance_results_full.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(RESULTS_DIR / "corpus_manifest_full.csv", index=False)

df_utt = pd.DataFrame(utt_rows)
df_utt.to_csv(RESULTS_DIR / "utterance_results_full.csv", index=False)
pd.DataFrame(manifest_rows).to_csv(RESULTS_DIR / "corpus_manifest_full.csv", index=False)

rows = []
for fl_cfg, w_lang, display, region in LANGUAGES:
    sub = df_utt[df_utt["fleurs_config"] == fl_cfg]
    if len(sub) == 0:
        continue
    mean_o, lo_o, hi_o = bootstrap_ci(sub["wer_ours"].values)
    mean_w, lo_w, hi_w = bootstrap_ci(sub["wer_whisper"].values)
    rows.append({
        "fleurs_config": fl_cfg, "display": display, "region": region,
        "n_utterances": len(sub),
        "total_audio_s": round(float(sub["duration_s"].sum()), 1),
        "mean_wer_ours": round(mean_o, 4),
        "wer_ours_ci95_lo": round(lo_o, 4), "wer_ours_ci95_hi": round(hi_o, 4),
        "mean_wer_whisper": round(mean_w, 4),
        "wer_whisper_ci95_lo": round(lo_w, 4), "wer_whisper_ci95_hi": round(hi_w, 4),
        "mean_lid_prob": round(float(sub["lid_prob"].mean()), 3),
    })
df_sum = pd.DataFrame(rows)
df_sum.to_csv(RESULTS_DIR / "per_language_summary_full.csv", index=False)
print(f"\n[save] {RESULTS_DIR / 'per_language_summary_full.csv'}")
print(df_sum.to_string(index=False))

en_vals = df_utt[df_utt["fleurs_config"] == "en_us"]["wer_whisper"].values
sig_rows = []
m = sum(1 for x in LANGUAGES if x[0] != "en_us")
for fl_cfg, w_lang, display, region in LANGUAGES:
    if fl_cfg == "en_us":
        continue
    sub = df_utt[df_utt["fleurs_config"] == fl_cfg]["wer_whisper"].values
    obs, p, lo, hi = unpaired_bootstrap_diff(en_vals, sub)
    sig_rows.append({
        "comparison": f"{fl_cfg} vs en_us",
        "mean_diff": round(obs, 4),
        "diff_ci95_lo": round(lo, 4), "diff_ci95_hi": round(hi, 4),
        "p_two_sided": round(p, 4),
        "p_bh_corrected": round(min(p * m, 1.0), 4),
    })
pd.DataFrame(sig_rows).to_csv(RESULTS_DIR / "pairwise_significance.csv", index=False)
print(f"[save] {RESULTS_DIR / 'pairwise_significance.csv'}")
print(pd.DataFrame(sig_rows).to_string(index=False))

prov = {
    "whisper_model": WHISPER_MODEL_SIZE,
    "whisper_device": WHISPER_DEVICE,
    "whisper_compute_type": WHISPER_COMPUTE,
    "beam_size": BEAM_SIZE,
    "n_max_per_lang": N_MAX_PER_LANG,
    "max_utt_seconds": MAX_UTT_SECONDS,
    "bootstrap_B": BOOTSTRAP_B,
    "rng_seed": RNG_SEED,
    "languages": [{"fleurs_config": l[0], "whisper_lang": l[1], "display": l[2]}
                  for l in LANGUAGES],
    "run_unix_ts": int(time.time()),
    "utterance_manifest": "results/corpus_manifest_full.csv",
    "normalisers": {
        "ours": "NFC + lowercase + punct-strip + ws-collapse",
        "whisper": "EnglishTextNormalizer (en); BasicTextNormalizer (all non-en), per Radford et al. 2023 Appendix D",
    },
}
(RESULTS_DIR / "run_provenance_full.json").write_text(json.dumps(prov, indent=2))
print(f"[save] {RESULTS_DIR / 'run_provenance_full.json'}")
print("\n[done] Full-split GPU pipeline complete.")
