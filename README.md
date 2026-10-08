# whisper-archival-bias

Replication code and summary results for the paper
**"What the Machine Cannot Hear: Evaluating Whisper on South Asian Languages for Diasporic Oral-History Archives"**
by Sayan Mandal and Jayosree Adhikari, accepted at the 2026 IEEE International Conference on Machine Learning and Applications (ICMLA 2026).

The paper evaluates OpenAI's Whisper on four South Asian languages (Hindi, Bengali, Tamil, Urdu) against an American-English baseline, on the openly licensed FLEURS speech corpus. It tests whether publicly available Indic-tuned Whisper-large-v2 fine-tunes close the per-language gap, extracts a deterministic word-level error typology, and anchors the FLEURS-derived numbers to a recent anecdotal Bengali recording of a Partition-era memory, shared with the authors under informed consent.

## Talk

The slides presented at ICMLA 2026 are in [`talk/whisper_archival_bias_talk.pdf`](talk/whisper_archival_bias_talk.pdf) (20 pages: 13 talk slides followed by backup slides on the protocol, the normaliser, worked error examples, limitations, references, and the Whisper architecture).

## Headline findings

| | English | Hindi | Bengali | Tamil | Urdu |
|---|---|---|---|---|---|
| Whisper-large-v3 WER (Whisper-Radford normaliser) | 0.042 | 0.161 | 0.464 | 0.210 | 0.215 |
| Ratio to English baseline | 1x | 3.8x | 11.0x | 5.0x | 5.1x |

Region-tuned comparator (deduplicated to one take per FLEURS sentence, both columns Whisper-Radford normalised on the same utterances):

| | n | Whisper-large-v3 | Indic fine-tune (large-v2) | Relative WER reduction |
|---|---|---|---|---|
| Hindi | 265 | 0.164 | 0.070 | 57% |
| Bengali | 319 | 0.466 | 0.213 | 54% |
| Tamil | 334 | 0.206 | 0.067 | 67% |

The fine-tunes reduce Indic WER by 54%-67% relative on read speech but do not close the gap to English (the Bengali fine-tune is still about 5x the English baseline). The Hindi and Tamil fine-tunes include FLEURS train and dev in their training mixture, so their reductions are an upper bound. No Urdu large-v2 fine-tune of comparable lineage was located.

On the anecdotal Bengali case-study clip, forcing the language hint to English produces 20 segments of fluent English at LID confidence 1.0 that mix accurate translation of the interview's Q&A opening with clearly hallucinated content. The model does not signal which spans are which.

All Indic-vs-English contrasts are significant at Benjamini-Hochberg adjusted q < 0.001 (the bootstrap floor at B=1000 resamples).

The word-level typology extracts 10,370 error events: character substitution 57.8%, matra or word-form error 18.8%, hallucinated insertion 13.5%, dropped non-English token 7.6%, script-to-anglicisation 1.2%, other deletion 1.1%.

FLEURS is read studio speech, the opposite register to spontaneous, code-switched diasporic audio, so these numbers are a floor.

## Repository layout

```
whisper-archival-bias/
├── README.md
├── requirements.txt
├── CITATION.cff
├── references.bib
├── docs/
│   └── REVIEWER_CHECKLIST.md         Full reviewer checklist per error mechanism
├── talk/
│   └── whisper_archival_bias_talk.pdf    Slides presented at ICMLA 2026 (with backup slides)
├── experiments/
│   ├── 03_run_full_whisper.py        Canonical Whisper-large-v3 evaluation
│   ├── 04_run_indic_comparator.py    Region-tuned Whisper-large-v2 comparator
│   ├── 05_run_case_study.py          Anecdotal case-study evaluation
│   ├── 06_make_figures_full.py       Figures, typology, LaTeX fragments
│   └── run_chain.sh                  Sequential runner for 04, 05, 06
├── figures/
│   ├── pipeline_architecture.pdf/.png    Fig. 1, evaluation pipeline
│   ├── wer_by_language.pdf               Fig. 2, WER by language with 95% CIs
│   ├── wer_distribution.pdf              Fig. 3, per-utterance WER distribution
│   ├── error_mechanism_stack.pdf         Fig. 4, error-mechanism composition
│   └── lid_vs_wer.pdf                    Supplementary: LID probability vs WER
└── results/
    ├── corpus_manifest_full.csv          Per-utterance manifest (FLEURS sample IDs, durations)
    ├── per_language_summary_full.csv     Table II: mean WER + bootstrap CI per language
    ├── per_language_summary_indic.csv    Table III: Indic fine-tunes vs matched canonical WER
    ├── pairwise_significance.csv         Bootstrap mean-difference vs English, BH q
    ├── error_type_counts.csv             Table IV / Fig. 4: word-level error counts by language
    ├── run_provenance_full.json          Canonical-pipeline configuration
    └── comparator_provenance.json        Indic-comparator configuration
```

The paper PDF and LaTeX source are distributed separately. This repository is the replication code, summary results, and the talk slides.

`06_make_figures_full.py` also writes the LaTeX table fragments used by the paper into `figures/*.tex`; these are regenerated on each run and not tracked.

The repository does not ship the raw FLEURS audio, the case-study audio, or the full per-utterance transcripts. Those artefacts are large, redundant with the open FLEURS dataset, and (for the case study) restricted by the narrator's consent terms. Re-running the scripts reproduces all transcripts and the full per-utterance CSV from scratch.

## Setup

The pipeline was developed on a single NVIDIA RTX 4070 Ti SUPER (16 GB). It also runs on smaller GPUs with `compute_type="int8_float16"` (see `experiments/03_run_full_whisper.py`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Reproduction

Run the four scripts in order. Step 1 is GPU-heavy (about 30-40 minutes on a 4070 Ti SUPER). Steps 2-4 take a few minutes each.

```bash
# 1. Canonical Whisper-large-v3 on FLEURS (5 langs, up to 600 utts each).
#    Produces results/utterance_results_full.csv and results/corpus_manifest_full.csv.
python experiments/03_run_full_whisper.py

# 2. Region-tuned Whisper-large-v2 Indic fine-tunes on the same audio.
#    Produces results/utterance_results_indic.csv.
python experiments/04_run_indic_comparator.py

# 3. Case study on the anecdotal Bengali clip (or any clip you supply).
#    The original clip is not shipped; pass the path to your own wav file.
python experiments/05_run_case_study.py path/to/clip.wav

# 4. Regenerate figures, the dual-WER summary CSVs, the error typology
#    (results/qualitative_errors.csv, results/error_type_counts.csv), and
#    all paper-included LaTeX fragments under figures/.
python experiments/06_make_figures_full.py
```

The chain runner waits for step 1 to finish and then runs 2-4 in sequence (it expects a `.venv` in the repository root, and step 3 needs the case-study clip at `experiments/audio_case_study/jjaudio_anecdote.wav`):

```bash
bash experiments/run_chain.sh
```

## Normalisation protocol

All Whisper-normalised WER values reported in the paper follow Radford et al. (2023): the `EnglishTextNormalizer` for English and the `BasicTextNormalizer` for non-English (Unicode NFC, lowercase, standard punctuation strip; no language-specific transliteration). We deliberately do not apply additional Indic-specific normalisers, because the Whisper paper's published per-language numbers were not computed under such normalisers.

One property of that protocol matters. The basic normaliser replaces Indic combining vowel signs with spaces, which splits words into base consonants: reference token counts rise by 1.8x for Hindi, 2.5x for Bengali and 3.4x for Tamil. Whisper-Radford WER for those languages is therefore not computed over words. We report it for comparability with published values, and also report an *Ours* WER (NFC, lowercase, punctuation strip, whitespace collapse) that preserves word boundaries. The word-level error typology runs on the *Ours* normalisation.

## Statistical testing

For each Indic-vs-English contrast we run an unpaired bootstrap of the mean WER difference (B=1000 resamples, seed 13). The two-sided p-value is floored at 1/B = 10^-3. The four contrasts are adjusted with the Benjamini-Hochberg step-up procedure.

## Reviewer checklist

`docs/REVIEWER_CHECKLIST.md` gives concrete reviewer actions for each error mechanism in the typology, for archivists reviewing Whisper transcripts of South Asian diasporic audio.

## Ethics and data terms

* FLEURS speakers consented to release under CC-BY 4.0.
* The case-study recording is an anecdotal personal collection by the authors. The narrator gave informed consent for academic use of the audio for the purposes of this paper, including quoting short excerpts of the model output. The narrator is anonymised; neither the audio file nor the model's full transcript of it (`results/case_study_anecdote.json`) is redistributed here. Name redactions for rendering are read from a local, untracked `experiments/redactions.local.txt`.
* SAADA and the 1947 Partition Archive reserve the right to grant written permission for any non-personal use of their recordings; the methodology in this paper is intended to extend to those archives only under their formal research-access protocols.

## Citation

If you use this code or the reviewer checklist, please cite:

```bibtex
@inproceedings{mandal2026machine,
  title     = {What the Machine Cannot Hear: Evaluating {Whisper} on {South Asian} Languages for Diasporic Oral-History Archives},
  author    = {Mandal, Sayan and Adhikari, Jayosree},
  booktitle = {Proceedings of the 2026 IEEE International Conference on Machine Learning and Applications (ICMLA)},
  year      = {2026}
}
```

## License

Code in this repository is released under the MIT License (see `LICENSE`). The paper itself is distributed separately; the case-study audio and the FLEURS corpus retain their own licensing terms.
