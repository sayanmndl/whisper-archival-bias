"""
Qualitative case study: Whisper-large-v3 (fp16) on an anecdotal personally
collected audio recording of a partition-era oral account (~98 s, Bengali
narration with possible English code-switch). The narrator gave informed
consent for academic use of the recording for this paper; the speaker is
anonymised. The raw audio file (jjaudio.mpeg) is held by the authors and
not redistributed; only this script's output is shipped.

We have no human reference transcript for this clip. Outputs are saved for
qualitative inspection (segment list with timestamps + auto-detected language
per segment) -- this anchors the FLEURS-derived numbers to an actual
diasporic oral-history clip.

Usage: python experiments/05_run_case_study.py [path/to/clip.wav]
(default: experiments/audio_case_study/jjaudio_anecdote.wav, not shipped)

Output: results/case_study_anecdote.json (gitignored; contains the narrator's
transcript and is not redistributed)
"""
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parents[1]
WAV = (Path(sys.argv[1]) if len(sys.argv) > 1
       else ROOT / "experiments" / "audio_case_study" / "jjaudio_anecdote.wav")
if not WAV.exists():
    sys.exit(f"[error] case-study audio not found: {WAV}. "
             "Pass the path to your own clip as the first argument.")
OUT = ROOT / "results" / "case_study_anecdote.json"

print("[init] Loading Whisper-large-v3 (cuda, float16)...")
t0 = time.time()
model = WhisperModel("large-v3", device="cuda", compute_type="float16")
print(f"[init] loaded in {time.time()-t0:.1f}s")

# Three runs: (a) no language hint -> Whisper's own LID; (b) language='bn'
# forces Bengali decoding; (c) language='en' forces English decoding.
results = {}
for label, kwargs in [
    ("no_lang_hint", dict(beam_size=5, vad_filter=True, condition_on_previous_text=False)),
    ("forced_bn",    dict(beam_size=5, vad_filter=True, condition_on_previous_text=False, language="bn")),
    ("forced_en",    dict(beam_size=5, vad_filter=True, condition_on_previous_text=False, language="en")),
]:
    print(f"\n[run] {label}")
    t = time.time()
    segs, info = model.transcribe(str(WAV), **kwargs)
    out_segs = []
    for s in segs:
        out_segs.append({
            "start": round(s.start, 2), "end": round(s.end, 2),
            "text": s.text.strip(),
        })
    results[label] = {
        "lid_detected": info.language,
        "lid_probability": float(info.language_probability),
        "all_lid_probs_top5": sorted(
            [(k, float(v)) for k, v in (info.all_language_probs or [])],
            key=lambda x: -x[1])[:5],
        "segments": out_segs,
        "decode_seconds": round(time.time() - t, 2),
    }
    print(f"  LID={info.language} p={info.language_probability:.3f} "
          f"segments={len(out_segs)} time={time.time()-t:.1f}s")

OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
print(f"\n[save] {OUT}")
print("[done] case study complete.")
