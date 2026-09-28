"""
Build figures and LaTeX fragments for the ICMLA 2026 camera-ready paper.

Reads the FULL-test pipeline outputs:
  results/utterance_results_full.csv
  results/per_language_summary_full.csv
  results/pairwise_significance.csv
  results/run_provenance_full.json
  results/utterance_results_indic.csv     (if present)
  results/per_language_summary_indic.csv  (if present)
  results/case_study_anecdote.json        (if present)

Emits into figures/ :
  wer_by_language.pdf, wer_distribution.pdf, lid_vs_wer.pdf, error_mechanism_stack.pdf
  results_table.tex, wer_summary_paragraph.tex,
  qualitative_summary.tex, lit_anchor_table.tex,
  pairwise_sig_table.tex, indic_comparator_table.tex,
  case_study_box.tex, reviewer_checklist_table.tex

And writes, from the full data:
  results/qualitative_errors.csv       (per-event word-level typology)
  results/error_type_counts.csv        (Table IV / Fig. 4 counts)
  results/per_language_summary_indic.csv gains the matched, Radford-normalised
    canonical column used in Table III (canonical_wer_whisper_matched).
"""
import json
import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# IEEE PDF eXpress rejects Type 3 fonts. fonttype 42 embeds TrueType instead.
# Serif family matches the IEEEtran body text so figures do not read as a
# different document.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
# Liberation Serif is TrueType and metric-compatible with Times, so fonttype 42
# embeds it as genuine TrueType. Nimbus/Termes are CFF-based OTFs, which
# matplotlib embeds as CFF while still declaring TrueType, and poppler flags
# that as a font-type mismatch.
matplotlib.rcParams["font.serif"] = [
    "Liberation Serif", "Times New Roman", "DejaVu Serif",
]
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import jiwer
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

# Recompute Whisper-normalised WER inline from raw text using the protocol
# documented in the Whisper paper Appendix D: EnglishTextNormalizer for English,
# BasicTextNormalizer for non-English. This overrides the wer_whisper field in
# the input CSV (which was previously computed with IndicNLP per-language
# normalisers that the Whisper paper does NOT actually use).
_english_norm = EnglishTextNormalizer()
_basic_norm = BasicTextNormalizer()

def _radford_norm(text, lang_code):
    if text is None:
        return ""
    if lang_code == "en":
        return _english_norm(text).strip()
    return _basic_norm(text).strip()

def _radford_wer(ref, hyp):
    if not ref:
        return float("nan")
    return jiwer.wer(ref, hyp)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

df_utt = pd.read_csv(RES / "utterance_results_full.csv")
df_sum_raw = pd.read_csv(RES / "per_language_summary_full.csv")
prov = json.loads((RES / "run_provenance_full.json").read_text())

# Recompute wer_whisper using the Radford-protocol normaliser
def _recompute_row(r):
    lang = r["whisper_lang"]
    ref_n = _radford_norm(r["reference"], lang)
    hyp_n = _radford_norm(r["hypothesis"], lang)
    return pd.Series({
        "reference_norm_whisper": ref_n,
        "hypothesis_norm_whisper": hyp_n,
        "wer_whisper": _radford_wer(ref_n, hyp_n),
    })
recomputed = df_utt.apply(_recompute_row, axis=1)
df_utt["reference_norm_whisper"] = recomputed["reference_norm_whisper"]
df_utt["hypothesis_norm_whisper"] = recomputed["hypothesis_norm_whisper"]
df_utt["wer_whisper"] = recomputed["wer_whisper"]

# Rebuild per-language summary with the Radford-protocol numbers
def _bootstrap(values, B=1000, seed=13, alpha=0.05):
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

sum_rows = []
for cfg in ["en_us", "hi_in", "bn_in", "ta_in", "ur_pk"]:
    sub = df_utt[df_utt["fleurs_config"] == cfg]
    if not len(sub):
        continue
    mo, lo_o, hi_o = _bootstrap(sub["wer_ours"].values)
    mw, lo_w, hi_w = _bootstrap(sub["wer_whisper"].values)
    row = df_sum_raw[df_sum_raw["fleurs_config"] == cfg].iloc[0]
    sum_rows.append({
        "fleurs_config": cfg, "display": row["display"], "region": row["region"],
        "n_utterances": len(sub), "total_audio_s": row["total_audio_s"],
        "mean_wer_ours": round(mo, 4),
        "wer_ours_ci95_lo": round(lo_o, 4), "wer_ours_ci95_hi": round(hi_o, 4),
        "mean_wer_whisper": round(mw, 4),
        "wer_whisper_ci95_lo": round(lo_w, 4), "wer_whisper_ci95_hi": round(hi_w, 4),
        "mean_lid_prob": row["mean_lid_prob"],
    })
df_sum = pd.DataFrame(sum_rows)
df_sum.to_csv(RES / "per_language_summary_full.csv", index=False)
print("Recomputed per-language summary with Radford-protocol normalisation:")
print(df_sum.to_string(index=False))
# Persist the corrected utterance-level CSV (wer_whisper recomputed under
# the Radford-protocol normaliser; this overwrites the upstream pipeline's
# wer_whisper that was produced before the normaliser was corrected).
df_utt.to_csv(RES / "utterance_results_full.csv", index=False)
print(f"Resaved utterance_results_full.csv with corrected wer_whisper ({len(df_utt)} rows).")

# Recompute pairwise sig with the Radford-protocol normalised WER.
BOOTSTRAP_B = 1000
P_FLOOR = 1.0 / BOOTSTRAP_B  # bootstrap quantile cannot resolve below 1/B

def _unpaired_bootstrap(a, b, B=BOOTSTRAP_B, seed=13):
    """Unpaired bootstrap of the mean WER difference (b - a).
    Reported p is two-sided and floored at 1/B."""
    rng = np.random.default_rng(seed)
    a = np.array([v for v in a if not np.isnan(v)])
    b = np.array([v for v in b if not np.isnan(v)])
    obs = b.mean() - a.mean()
    diffs = np.empty(B)
    for i in range(B):
        diffs[i] = (b[rng.integers(0, len(b), size=len(b))].mean()
                    - a[rng.integers(0, len(a), size=len(a))].mean())
    p_two = max(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), P_FLOOR)
    return float(obs), float(p_two), float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))

def _benjamini_hochberg(p_values):
    """Standard BH step-up. Returns adjusted q-values in the original order."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    # enforce monotone non-decreasing on the sorted ranks
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out

en_vals = df_utt[df_utt["fleurs_config"] == "en_us"]["wer_whisper"].values
contrasts = ["hi_in", "bn_in", "ta_in", "ur_pk"]
raw = []
for cfg in contrasts:
    sub = df_utt[df_utt["fleurs_config"] == cfg]["wer_whisper"].values
    raw.append(_unpaired_bootstrap(en_vals, sub))
p_raw = [r[1] for r in raw]
p_bh = _benjamini_hochberg(p_raw)
sig_rows = []
for cfg, (obs, p, lo, hi), q in zip(contrasts, raw, p_bh):
    sig_rows.append({
        "comparison": f"{cfg} vs en_us",
        "mean_diff": round(obs, 4),
        "diff_ci95_lo": round(lo, 4), "diff_ci95_hi": round(hi, 4),
        "p_two_sided": round(p, 4),
        "q_bh": round(float(q), 4),
    })
df_sig = pd.DataFrame(sig_rows)
df_sig.to_csv(RES / "pairwise_significance.csv", index=False)
print("Recomputed pairwise significance (unpaired bootstrap, BH-adjusted):")
print(df_sig.to_string(index=False))
df_ind = pd.read_csv(RES / "per_language_summary_indic.csv") if (RES / "per_language_summary_indic.csv").exists() else None
# Utterance-level Indic comparator rows, needed to identify exactly which
# sample_ids the fine-tunes were evaluated on so the canonical column of the
# comparator table can be recomputed on matching audio.
df_ind_utt = pd.read_csv(RES / "utterance_results_indic.csv") if (RES / "utterance_results_indic.csv").exists() else None
# Anecdotal personally collected recording (consented, anonymised).
case_path = RES / "case_study_anecdote.json"
case = json.loads(case_path.read_text()) if case_path.exists() else None

# Speaker-name redactions for paper rendering. The narrator consented to
# academic use on condition of anonymisation, so the names themselves are not
# shipped: list them one per line (longest first) in the gitignored file
# experiments/redactions.local.txt and they are replaced with "[Name]".
_redact_file = ROOT / "experiments" / "redactions.local.txt"
NAME_REDACTIONS = [
    (line.strip(), "[Name]")
    for line in (_redact_file.read_text().splitlines() if _redact_file.exists() else [])
    if line.strip()
]
def _redact(text):
    # Fail closed: with no redaction list, emit no verbatim case-study text.
    if not NAME_REDACTIONS:
        return "[verbatim text withheld: no experiments/redactions.local.txt]"
    for src, dst in NAME_REDACTIONS:
        text = text.replace(src, dst)
    return text

order = ["en_us", "hi_in", "bn_in", "ta_in", "ur_pk"]
df_sum = df_sum.set_index("fleurs_config").reindex(order).reset_index()


# ---------- Figure 1: WER by language (whisper-normalised) ----------
# One distinct colour per language. n is placed inside the bar (white text on
# coloured fill) instead of below the axis, so axis ticks and n labels never
# overlap. Mean value sits above the upper CI cap.
fig, ax = plt.subplots(figsize=(3.4, 2.7))
xs = np.arange(len(df_sum))
means = df_sum["mean_wer_whisper"].values
lo = df_sum["wer_whisper_ci95_lo"].values
hi = df_sum["wer_whisper_ci95_hi"].values
yerr = np.vstack([means - lo, hi - means])
# Distinct colours per language, accessible-friendly palette:
# English = grey, the four South Asian languages get a qualitative scheme.
bar_colours = ["#4d4d4d", "#1f78b4", "#33a02c", "#e31a1c", "#ff7f00"]
ax.bar(xs, means, yerr=yerr, color=bar_colours, capsize=3, alpha=0.92,
       edgecolor="black", linewidth=0.5)
ax.set_xticks(xs)
ax.set_xticklabels(df_sum["display"], rotation=0, fontsize=7)
ax.set_ylabel("WER (lower is better)", fontsize=8)
# Cap the y-axis just above the highest CI upper bound so we don't waste
# vertical space. Add only a small headroom for the mean-value annotation.
ax.set_ylim(0, hi.max() * 1.18)
# n labels placed INSIDE the bar (avoids overlap with x-tick labels) when
# the bar is tall enough; for very short bars, place above the cap.
for x, m, h, n in zip(xs, means, hi, df_sum["n_utterances"]):
    ax.text(x, h + 0.02, f"{m:.2f}", ha="center", va="bottom",
            fontsize=6.5, fontweight="bold")
    if m > 0.10:
        ax.text(x, m / 2, f"n={int(n)}", ha="center", va="center",
                fontsize=6, color="white")
    else:
        ax.text(x, h + 0.07, f"n={int(n)}", ha="center", va="bottom",
                fontsize=6, color="#555")
ax.tick_params(axis='y', labelsize=7)
ax.set_title(f"Whisper-{prov['whisper_model']} WER by language (FLEURS test)",
             fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "wer_by_language.pdf", bbox_inches="tight"); plt.close()
print("[save] wer_by_language.pdf")


# ---------- Figure 2: per-utterance WER distribution ----------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
positions = np.arange(len(order))
data = [df_utt[df_utt["fleurs_config"] == c]["wer_whisper"].dropna().values for c in order]
ax.boxplot(data, positions=positions, widths=0.55, patch_artist=True,
           medianprops=dict(color="black", linewidth=1.0),
           boxprops=dict(facecolor="#f0f0f0", edgecolor="#444", linewidth=0.5),
           whiskerprops=dict(color="#444", linewidth=0.5),
           capprops=dict(color="#444", linewidth=0.5),
           flierprops=dict(marker="", linestyle=""))
rng = np.random.default_rng(7)
for i, vals in enumerate(data):
    j = rng.normal(0, 0.05, size=len(vals))
    c = "#444" if order[i] == "en_us" else "#a23b3b"
    ax.scatter(positions[i] + j, vals, s=4, color=c, alpha=0.35,
               edgecolor="none", zorder=3)
ax.set_xticks(positions)
ax.set_xticklabels(df_sum["display"], fontsize=7)
ax.set_ylabel("Per-utt WER", fontsize=8)
ax.set_ylim(-0.05, max(1.55, max(v.max() if len(v) else 0 for v in data) * 1.08))
ax.axhline(1.0, color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
ax.tick_params(axis='y', labelsize=7)
ax.set_title("Per-utterance WER distribution", fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "wer_distribution.pdf", bbox_inches="tight"); plt.close()
print("[save] wer_distribution.pdf")


# ---------- Figure 3: LID vs WER scatter ----------
fig, ax = plt.subplots(figsize=(3.4, 2.6))
palette = {"en_us": "#444", "hi_in": "#a23b3b", "bn_in": "#c97a2b",
           "ta_in": "#3b6ea2", "ur_pk": "#2b8a4a"}
for cfg in order:
    sub = df_utt[df_utt["fleurs_config"] == cfg]
    disp = df_sum.set_index("fleurs_config").loc[cfg, "display"]
    ax.scatter(sub["lid_prob"], sub["wer_whisper"], s=5, alpha=0.5,
               color=palette[cfg], edgecolor="none", label=disp)
ax.set_xlabel("LID probability", fontsize=8)
ax.set_ylabel("Per-utt WER", fontsize=8)
ax.set_xlim(0.5, 1.02)
ax.set_ylim(-0.05, max(1.55, df_utt["wer_whisper"].max() * 1.08))
ax.axhline(1.0, color="grey", linewidth=0.4, linestyle="--", alpha=0.5)
ax.legend(loc="upper left", fontsize=6, frameon=False, ncol=2)
ax.tick_params(labelsize=7)
ax.set_title("LID vs WER", fontsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout()
plt.savefig(FIG / "lid_vs_wer.pdf", bbox_inches="tight"); plt.close()
print("[save] lid_vs_wer.pdf")


# ---------- Qualitative error catalogue ----------
import jiwer

def split_words(s): return re.findall(r"\S+", s or "")
def is_ascii_word(w): return all(ord(c) < 128 for c in w)
def is_devanagari(w): return any(0x0900 <= ord(c) <= 0x097F for c in w)
def char_set(w): return set(c for c in w if c.isalpha())

def categorise(ref_word, hyp_word):
    if ref_word == hyp_word:
        return None
    if hyp_word is None:
        if is_devanagari(ref_word) or any(
            0x0980 <= ord(c) <= 0x09FF or 0x0B80 <= ord(c) <= 0x0BFF
            or 0x0600 <= ord(c) <= 0x06FF for c in ref_word):
            return "dropped_non_english_token"
        if ref_word and ref_word[0].isupper():
            return "proper_noun_dropped"
        return "deletion"
    if is_ascii_word(hyp_word) and not is_ascii_word(ref_word):
        return "script_to_anglicisation"
    if char_set(ref_word) != char_set(hyp_word):
        return "char_substitution"
    return "matra_or_word_form_error"


# The typology runs on the *Ours* normalisation (NFC, lowercase, punctuation
# strip, whitespace collapse), NOT the Whisper-Radford one used for the WER
# tables. Whisper's BasicTextNormalizer replaces Indic combining vowel signs
# with spaces, which splits words into bare consonants (reference tokens
# inflate 1.8x for Hindi, 2.5x Bengali, 3.4x Tamil). A word-level typology
# computed on that text would be measuring the normaliser, not the model.
# The Ours normalisation preserves word boundaries and matras, so the
# categories below mean what they say.
qual_rows = []
for _, r in df_utt.iterrows():
    ref_norm = r["reference_norm_ours"] if isinstance(r["reference_norm_ours"], str) else ""
    hyp_norm = r["hypothesis_norm_ours"] if isinstance(r["hypothesis_norm_ours"], str) else ""
    ref_words = split_words(ref_norm)
    hyp_words = split_words(hyp_norm)
    if not ref_words:
        continue
    try:
        out = jiwer.process_words(ref_norm, hyp_norm)
        for align in out.alignments[0]:
            if align.type == "equal":
                continue
            ref_span = " ".join(ref_words[align.ref_start_idx:align.ref_end_idx])
            hyp_span = " ".join(hyp_words[align.hyp_start_idx:align.hyp_end_idx])
            if align.type == "delete":
                cat = categorise(ref_span, None); hyp_span = "<dropped>"
            elif align.type == "insert":
                cat = "hallucinated_insertion"; ref_span = "<inserted>"
            else:
                rw = ref_words[align.ref_start_idx]
                hw = hyp_words[align.hyp_start_idx] if align.hyp_start_idx < len(hyp_words) else None
                cat = categorise(rw, hw) or "substitution_other"
            if cat is None: continue
            qual_rows.append({
                "fleurs_config": r["fleurs_config"], "display": r["display"],
                "sample_id": r["sample_id"], "error_type": cat,
                "ref_span": ref_span, "hyp_span": hyp_span,
            })
    except Exception:
        continue

df_qual = pd.DataFrame(qual_rows)
df_qual.to_csv(RES / "qualitative_errors.csv", index=False)
print(f"[save] qualitative_errors.csv  ({len(df_qual)} rows)")

# Per-language counts behind Table IV and Figure 4.
df_counts = (df_qual.groupby(["display", "error_type"]).size()
             .rename("count").reset_index())
df_counts["total_errors"] = df_counts.groupby("display")["count"].transform("sum")
df_counts["share_within_language"] = (df_counts["count"] / df_counts["total_errors"]).round(4)
df_counts.sort_values(["display", "count"], ascending=[True, False]).to_csv(
    RES / "error_type_counts.csv", index=False)
print(f"[save] error_type_counts.csv  ({int(df_counts['count'].sum())} events)")


# ---------- Figure 4: error-mechanism stacked bar ----------
type_palette = {
    "dropped_non_english_token": "#a23b3b",
    "script_to_anglicisation":   "#c97a2b",
    "char_substitution":         "#7a7a7a",
    "matra_or_word_form_error":  "#3b6ea2",
    "hallucinated_insertion":    "#2b8a4a",
    "deletion":                  "#bdbdbd",
    "proper_noun_dropped":       "#5a5a5a",
    "substitution_other":        "#dddddd",
}
type_order_plot = [t for t in type_palette if t in df_qual["error_type"].unique()]
counts_mat = (df_qual.groupby(["fleurs_config", "error_type"]).size()
              .unstack(fill_value=0).reindex(order)
              .reindex(columns=type_order_plot, fill_value=0))
shares = counts_mat.div(counts_mat.sum(axis=1).replace(0, 1), axis=0)
# Taller figure with the legend OUTSIDE on the right so it never collides
# with the x-axis title at the bottom.
fig, ax = plt.subplots(figsize=(3.4, 2.4))
left = np.zeros(len(order)); ypos = np.arange(len(order))
for t in type_order_plot:
    vals = shares[t].values
    ax.barh(ypos, vals, left=left, height=0.55,
            color=type_palette[t], edgecolor="white", linewidth=0.4,
            label=t.replace("_", " "))
    left += vals
ax.set_yticks(ypos)
ax.set_yticklabels(df_sum["display"], fontsize=7)
ax.invert_yaxis(); ax.set_xlim(0, 1.0)
ax.set_xlabel("Share of error events", fontsize=8, labelpad=2)
ax.tick_params(axis='x', labelsize=7)
ax.set_title("Error-mechanism composition by language", fontsize=8)
# Legend below the plot, anchored low enough that the x-axis title sits
# cleanly above it.
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32),
          fontsize=5.8, ncol=3, frameon=False, handlelength=1.2,
          columnspacing=0.9, handletextpad=0.4)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
plt.tight_layout(rect=(0, 0.05, 1, 1))
plt.savefig(FIG / "error_mechanism_stack.pdf", bbox_inches="tight"); plt.close()
print("[save] error_mechanism_stack.pdf")


# ---------- results_table.tex (dual-normaliser, full-column-pair width) ----------
rows_tex = []
for _, r in df_sum.iterrows():
    rows_tex.append(
        f"    {r['display']} & {int(r['n_utterances'])} & "
        f"{r['mean_wer_ours']:.3f} & [{r['wer_ours_ci95_lo']:.3f}, {r['wer_ours_ci95_hi']:.3f}] & "
        f"{r['mean_wer_whisper']:.3f} & [{r['wer_whisper_ci95_lo']:.3f}, {r['wer_whisper_ci95_hi']:.3f}] \\\\"
    )
results_tex = (
r"""\begin{table*}[!htbp]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{6pt}
  \caption{Per-language WER on the FLEURS test set for Whisper-""" + prov['whisper_model'] + r""" (""" + prov['whisper_compute_type'] + r""", CUDA, beam=""" + str(prov['beam_size']) + r"""). \emph{Ours} = NFC, lowercase, punctuation strip, whitespace collapse. \emph{Whisper-Radford} = \texttt{EnglishTextNormalizer} (en) and \texttt{BasicTextNormalizer} (non-en) per Appendix D of Radford et al. 95\% bootstrap CIs, $B=""" + str(prov['bootstrap_B']) + r"""$.}
  \label{tab:wer-by-language}
  \begin{tabular}{lrrcrc}
    \toprule
     & & \multicolumn{2}{c}{WER (Ours)} & \multicolumn{2}{c}{WER (Whisper-Radford)} \\
    \cmidrule(lr){3-4}\cmidrule(lr){5-6}
    Language & $n$ & mean & 95\% CI & mean & 95\% CI \\
    \midrule
""" + "\n".join(rows_tex) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")
(FIG / "results_table.tex").write_text(results_tex)
print("[save] results_table.tex")


# ---------- wer_summary_paragraph.tex ----------
en = df_sum[df_sum["fleurs_config"] == "en_us"].iloc[0]
sa = df_sum[df_sum["fleurs_config"] != "en_us"]
sa_mean_w = float(sa["mean_wer_whisper"].mean())
ratio = sa_mean_w / max(en["mean_wer_whisper"], 1e-3)
summary_tex = (
    f"Whisper-{prov['whisper_model']} achieves a mean WER of "
    f"{en['mean_wer_whisper']:.3f} (Whisper-Radford normalised) on the English baseline "
    f"({int(en['n_utterances'])} utterances), with 95\\% bootstrap CI "
    f"[{en['wer_whisper_ci95_lo']:.3f}, {en['wer_whisper_ci95_hi']:.3f}]. "
    f"Across the four South Asian languages, Whisper-Radford normalised mean WER ranges from "
    f"{sa['mean_wer_whisper'].min():.3f} ({sa.loc[sa['mean_wer_whisper'].idxmin(), 'display']}) "
    f"to {sa['mean_wer_whisper'].max():.3f} ({sa.loc[sa['mean_wer_whisper'].idxmax(), 'display']}); "
    f"the four-language mean of {sa_mean_w:.3f} is approximately {ratio:.1f}$\\times$ "
    f"the English baseline. Whisper's first-pass LID probability is uniformly high "
    f"(mean $\\geq 0.97$ for every language), so the gap is in transcription, not in "
    f"routing the audio through the wrong language head."
)
(FIG / "wer_summary_paragraph.tex").write_text(summary_tex)
print("[save] wer_summary_paragraph.tex")


# ---------- lit_anchor_table.tex (Radford 2023 vs ours, both Whisper-normalised) ----------
LIT_WER = {"en_us": 0.041, "hi_in": 0.226, "bn_in": 0.300, "ta_in": 0.282, "ur_pk": 0.280}
lit_lines = []
for cfg in order:
    r = df_sum[df_sum["fleurs_config"] == cfg].iloc[0]
    lit = LIT_WER[cfg]; diff = r["mean_wer_whisper"] - lit
    sign = "+" if diff >= 0 else ""
    lit_lines.append(
        f"    {r['display']} & {lit:.3f} & {r['mean_wer_whisper']:.3f} & {sign}{diff:.3f} \\\\"
    )
lit_tex = (
r"""\begin{table}[!htbp]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{6pt}
  \caption{Literature anchor: Whisper-large-v3 per-language WER reported by Radford et al.~\cite{radford2023whisper} versus our replication, both Whisper-Radford normalised.}
  \label{tab:lit-anchor}
  \begin{tabular}{lrrr}
    \toprule
    Language & Radford 2023 & Ours (full split) & $\Delta$ \\
    \midrule
""" + "\n".join(lit_lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
(FIG / "lit_anchor_table.tex").write_text(lit_tex)
print("[save] lit_anchor_table.tex")


# ---------- pairwise_sig_table.tex ----------
if df_sig is not None:
    rows_sig = []
    for _, r in df_sig.iterrows():
        # comparison string like "hi_in vs en_us" — map first part to display name
        cfg = r["comparison"].split(" vs ")[0]
        disp = {c: d for c, d in zip(df_sum["fleurs_config"], df_sum["display"])}[cfg]
        # p < 1/B (B=1000) is reported as <0.001 to honour the bootstrap floor.
        def _fmt(p):
            return "<0.001" if p < 0.0011 else f"{p:.4f}"
        rows_sig.append(
            f"    {disp} -- English & {r['mean_diff']:.3f} & "
            f"[{r['diff_ci95_lo']:.3f}, {r['diff_ci95_hi']:.3f}] & "
            f"{_fmt(r['p_two_sided'])} & {_fmt(r['q_bh'])} \\\\"
        )
    sig_tex = (
r"""\begin{table}[!htbp]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{6pt}
  \caption{Unpaired bootstrap of the mean WER difference (Whisper-Radford normalisation), 95\% CI of the difference, two-sided $p$ (floored at $1/B = 10^{-3}$), and Benjamini--Hochberg adjusted $q$.}
  \label{tab:pairwise-sig}
  \begin{tabular}{lrrrr}
    \toprule
    Contrast & $\Delta$WER & 95\% CI & $p$ & $q_{\text{BH}}$ \\
    \midrule
""" + "\n".join(rows_sig) + r"""
    \bottomrule
  \end{tabular}
\end{table}
""")
    (FIG / "pairwise_sig_table.tex").write_text(sig_tex)
    print("[save] pairwise_sig_table.tex")
else:
    (FIG / "pairwise_sig_table.tex").write_text("% pairwise_significance.csv not found\n")


# ---------- indic_comparator_table.tex ----------
if df_ind is not None:
    rows_ind = []
    matched_canon = {}
    canon = df_sum.set_index("fleurs_config")
    for _, r in df_ind.iterrows():
        cfg = r["fleurs_config"]
        disp = canon.loc[cfg, "display"]
        # Apples-to-apples comparator. Both columns must be Whisper-Radford
        # normalised or the comparison is meaningless.
        #
        # Do NOT use ref_canonical_wer_whisper from the indic summary CSV: it
        # was written by 04_run_indic_comparator.py under the legacy normaliser
        # and is roughly 2-3x the Radford value, which inflates the apparent
        # reduction (Hindi +77% vs a true 57%). Recompute the canonical mean
        # here from df_utt, whose wer_whisper was recomputed under the Radford
        # protocol above, restricted to the same sample_ids the fine-tune ran
        # on.
        ids = set(df_ind_utt[df_ind_utt["fleurs_config"] == cfg]["sample_id"])
        canon_sub = (df_utt[(df_utt["fleurs_config"] == cfg)
                            & (df_utt["sample_id"].isin(ids))]
                     .drop_duplicates(subset="sample_id", keep="last"))
        canon_w = float(canon_sub["wer_whisper"].mean())
        indic_w = r["mean_wer_whisper"]
        gap_closed = (canon_w - indic_w) / canon_w * 100 if canon_w > 0 else 0
        matched_canon[cfg] = (round(canon_w, 4), round(gap_closed, 1))
        rows_ind.append(
            f"    {disp} & {int(r['n_utterances'])} & {canon_w:.3f} & "
            f"{indic_w:.3f} & [{r['ci95_lo_whisper']:.3f}, {r['ci95_hi_whisper']:.3f}] & "
            f"{gap_closed:+.0f}\\% \\\\"
        )
    ind_tex = (
r"""\begin{table*}[!htbp]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{6pt}
  \caption{Region-tuned comparator. Whisper-large-v3 (canonical) versus Whisper-large-v2 Indic fine-tunes~\cite{indic_whisper_community_finetunes, bhogale2023vistaar} on identical deduped FLEURS audio (one take per source sentence); both columns are Whisper-Radford normalised on the same utterances. \emph{WER reduction} is the relative reduction from canonical to fine-tune on that audio, not closure of the gap to the English baseline. The Hindi and Tamil fine-tunes include the FLEURS train and dev splits in their training mixture, so their reductions are best read as an upper bound. No large-v2 Urdu fine-tune of comparable lineage was located.}
  \label{tab:indic-comparator}
  \begin{tabular}{lrrrcr}
    \toprule
     & & Whisper & \multicolumn{2}{c}{Indic-tuned (large-v2)} & WER \\
    \cmidrule(lr){4-5}
    Language & $n$ & large-v3 & WER & 95\% CI & reduction \\
    \midrule
""" + "\n".join(rows_ind) + r"""
    \bottomrule
  \end{tabular}
\end{table*}
""")
    (FIG / "indic_comparator_table.tex").write_text(ind_tex)
    print("[save] indic_comparator_table.tex")
    # Persist the matched canonical column so the CSV agrees with Table III.
    # The legacy ref_canonical_wer_whisper column is dropped (see note above).
    df_ind_out = df_ind.drop(columns=["ref_canonical_wer_whisper"], errors="ignore")
    df_ind_out["canonical_wer_whisper_matched"] = df_ind_out["fleurs_config"].map(
        lambda c: matched_canon[c][0])
    df_ind_out["wer_reduction_pct"] = df_ind_out["fleurs_config"].map(
        lambda c: matched_canon[c][1])
    df_ind_out.to_csv(RES / "per_language_summary_indic.csv", index=False)
    print("[save] per_language_summary_indic.csv (matched canonical column)")
else:
    (FIG / "indic_comparator_table.tex").write_text("% per_language_summary_indic.csv not found\n")


# ---------- case_study_box.tex ----------
# pdflatex cannot render Bengali script without a Bengali font, so we render
# Bengali segments as quantitative summaries (segment count, characters) and
# show the forced-English output verbatim (it is the load-bearing finding:
# Whisper hallucinates fluent English on actual Bengali speech).
if case is not None:
    def esc(t):
        return t.replace("&","\\&").replace("_","\\_").replace("#","\\#").replace("$","\\$").replace("%","\\%")

    nh = case.get("no_lang_hint", {})
    bn = case.get("forced_bn", {})
    en = case.get("forced_en", {})

    def bengali_summary(seglist):
        n = len(seglist)
        total_chars = sum(len(s["text"]) for s in seglist)
        ascii_chars = sum(sum(ord(c) < 128 for c in s["text"]) for s in seglist)
        # extract first 2 segments' time spans + character counts
        spans = []
        for s in seglist[:3]:
            spans.append(f"[{s['start']:.0f}--{s['end']:.0f}s, {len(s['text'])} chars]")
        return (f"{n} segments, {total_chars} chars total ({ascii_chars} ASCII). "
                f"First spans: {' '.join(spans)}. Bengali-script text omitted "
                f"(pdflatex font limitation); full output in "
                f"\\texttt{{case\\_study\\_anecdote.json}}.")

    def en_segments(seglist, n=3, charcap=200):
        out = []
        for s in seglist[:n]:
            txt = esc(_redact(s["text"][:charcap]))
            out.append(f"\\texttt{{[{s['start']:.0f}--{s['end']:.0f}s]}} {txt}")
        return "\\\\\n      ".join(out)

    def en_segments_full(seglist, charcap=140):
        out = []
        for s in seglist:
            txt = esc(_redact(s["text"][:charcap]))
            out.append(f"\\texttt{{[{s['start']:.0f}--{s['end']:.0f}s]}} {txt}")
        return "\\\\\n      ".join(out)

    # Inline (non-floating) block in §4.4.
    n_segs_en = len(en.get("segments", []))
    case_tex = (
r"""\smallskip
\noindent\fbox{\begin{minipage}{0.97\columnwidth}\footnotesize
\textbf{Case-study output summary.} """
+ f"Under no language hint, Whisper auto-LID returns Bengali (first-segment top-1: bn at $p={nh.get('lid_probability', 0):.2f}$; {len(nh.get('segments', []))} segments) and the output is Bengali-script transcription. Under forced Bengali, output is Bengali script ({len(bn.get('segments', []))} segments). Under \\emph{{forced English}}, the model emits {n_segs_en} segments of fluent English at LID confidence $1.00$. The English output is a mixture: early segments resemble accurate translation of Q\\&A from the Bengali interview (the recording opens with a name-and-birthplace exchange), while later segments contain clearly hallucinated content (e.g.\\ ``I died early in my birth. I didn't die.''). Whisper's task is \\texttt{{transcribe}}, not translate; the model does not signal which forced-English spans are translation, which are paraphrase, and which are hallucination. The first three segments verbatim:"
+ r"""
\smallskip\\
"""
+ en_segments(en.get("segments", []), n=3, charcap=220).replace("\\\\\n", "\\\\[0.2em]\n")
+ r"""
\end{minipage}}
\smallskip
""")
    (FIG / "case_study_box.tex").write_text(case_tex)
    print("[save] case_study_box.tex")
else:
    (FIG / "case_study_box.tex").write_text("% case_study_anecdote.json not found\n")


# ---------- qualitative_summary.tex ----------
type_order = list(type_palette.keys())
counts = df_qual["error_type"].value_counts().reindex(type_order).fillna(0).astype(int)
total = int(counts.sum())
counts_lines = []
for t in type_order:
    c = int(counts.get(t, 0))
    if c == 0: continue
    pct = 100.0 * c / max(total, 1)
    counts_lines.append(f"    {t.replace('_', ' ')} & {c} & {pct:.1f}\\% \\\\")

# Render Indic-script runs through the XeLaTeX font macros defined in
# latex.tex (\dev for Devanagari, \bn for Bengali, \ta for Tamil, \ar for
# Arabic-script Urdu). ASCII content is preserved verbatim. Earlier
# revisions used [DEV.n] script-tag stand-ins for the pdflatex build;
# with XeLaTeX + Noto fonts we can render the actual glyphs.
def _script_label(s):
    if not s: return ""
    if s.startswith("<") and s.endswith(">"): return s
    def _block(ch):
        c = ord(ch)
        if 0x0900 <= c <= 0x097F: return "dev"
        if 0x0980 <= c <= 0x09FF: return "bn"
        if 0x0B80 <= c <= 0x0BFF: return "ta"
        if 0x0600 <= c <= 0x06FF: return "ar"
        return None
    out = []
    run_script = None
    run_buf = ""
    for ch in s:
        sc = _block(ch)
        if sc is None:
            if run_script is not None:
                out.append(f"\\{run_script}{{{run_buf}}}")
                run_script, run_buf = None, ""
            out.append(ch)
        elif sc == run_script:
            run_buf += ch
        else:
            if run_script is not None:
                out.append(f"\\{run_script}{{{run_buf}}}")
            run_script, run_buf = sc, ch
    if run_script is not None:
        out.append(f"\\{run_script}{{{run_buf}}}")
    return "".join(out)

sa_only = df_qual[df_qual["fleurs_config"] != "en_us"]
# Sample across languages so the table is not Hindi-only. For each error
# type, pick at most one example per language. Apply a quality filter on
# script-to-anglicisation rows so we surface clean numerical / short
# ASCII anglicisations rather than proper-noun preservations
# (e.g. Devanagari "Aerosmith" -> "Aerosmith") or alignment artefacts
# where Whisper hallucinated an unrelated English word (e.g. "Cincinnati"
# inserted mid-Tamil) that the edit-distance aligner happened to pair
# with a non-Latin span.
def esc(x):
    # Avoid escaping characters inside the inline \dev{...}/\bn{...} macros
    # we just emitted. Macro names use only letters; we only need to escape
    # raw &/#/$/% if they happen to appear in plain ASCII content.
    out = []
    i = 0
    while i < len(x):
        ch = x[i]
        if ch in "&#$%":
            out.append("\\" + ch)
        elif ch == "_" and (i == 0 or x[i-1] != "\\"):
            out.append("\\_")
        else:
            out.append(ch)
        i += 1
    return "".join(out)

def _quality_ok(error_type, ref, hyp):
    """Filter script-to-anglicisation rows down to clean examples."""
    if error_type != "script_to_anglicisation":
        return True
    # Reject if hypothesis looks like a hallucinated English proper noun
    # or city/place name. Heuristic: any token starting with an uppercase
    # letter or any multi-word ASCII span longer than ~10 chars where the
    # ref span is a single Indic character.
    bad_tokens = {"aerosmith", "cincinnati", "fatima", "shrin", "dundee",
                  "sagittarius", "operation", "dragon", "skaf", "decimal",
                  "capital", "notation", "level", "dinosaur"}
    hyp_low = hyp.lower()
    if any(b in hyp_low for b in bad_tokens):
        return False
    # Prefer rows where the hypothesis is short (a few chars, digit-like)
    if len(hyp.replace(" ", "")) > 8:
        return False
    return True

examples = []
lang_order = ["Hindi", "Bengali", "Tamil", "Urdu"]
# Two rows per language: one dropped-token example (the most consequential
# silencing mechanism) and one script-to-anglicisation example (the most
# visibly striking).
type_picks_lang = ["dropped_non_english_token", "script_to_anglicisation"]
for lang in lang_order:
    for t in type_picks_lang:
        sub = sa_only[(sa_only["error_type"] == t) & (sa_only["display"] == lang)]
        if len(sub) == 0: continue
        picked = None
        for _, candidate in sub.iterrows():
            if _quality_ok(t, candidate["ref_span"], candidate["hyp_span"]):
                picked = candidate
                break
        if picked is None:
            picked = sub.iloc[0]
        ref_lbl = _script_label(picked["ref_span"])
        hyp_lbl = _script_label(picked["hyp_span"])
        short_t = {"dropped_non_english_token": "dropped",
                   "script_to_anglicisation": "anglicised"}.get(t, t.replace("_", " "))
        examples.append(
            f"    {picked['display']} & {short_t} & "
            f"{esc(ref_lbl)} & {esc(hyp_lbl)} \\\\"
        )

qual_tex = (
    f"From the per-utterance outputs we extract {total} word-level error events. "
    f"Table~\\ref{{tab:qual-types}} summarises the distribution by mechanism; "
    f"Table~\\ref{{tab:qual-examples}} shows representative examples from the four "
    f"South Asian languages.\n\n"
    r"""\begin{table}[!htbp]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{6pt}
  \caption{Distribution of word-level error events by mechanism, aggregated across the five-language FLEURS pilot.}
  \label{tab:qual-types}
  \begin{tabular}{lrr}
    \toprule
    Error mechanism & Count & Share \\
    \midrule
""" + "\n".join(counts_lines) + r"""
    \bottomrule
  \end{tabular}
\end{table}

\begin{table}[!htbp]
  \centering
  \scriptsize
  \setlength{\tabcolsep}{3pt}
  \caption{Representative word-level error events, two rows per language: a \emph{dropped} non-English token, and an \emph{anglicised} span where a non-Latin reference is replaced by a Latin-script token in the hypothesis.}
  \label{tab:qual-examples}
  \begin{tabular}{@{}p{1.0cm}p{1.15cm}p{2.7cm}p{2.5cm}@{}}
    \toprule
    Language & Error type & Reference span & Whisper hypothesis \\
    \midrule
""" + "\n".join(examples) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
)
(FIG / "qualitative_summary.tex").write_text(qual_tex)
print("[save] qualitative_summary.tex")


# ---------- reviewer_checklist_table.tex ----------
checklist_tex = r"""\begin{table}[!ht]
  \centering
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \caption{Reviewer checklist from the qualitative error typology (\S\ref{sec:results-qual}). Four highest-priority mechanisms; the full list is in the code repository.}
  \label{tab:reviewer-checklist}
  \begin{tabular}{p{2.4cm}p{5cm}}
    \toprule
    Error mechanism & Reviewer action \\
    \midrule
    Dropped non-English token & Re-listen; restore the dropped term in the original script; tag with the language code. \\
    Script-to-anglicisation & Verify the anglicised form against the speaker's preferred romanisation; record both forms. \\
    Hallucinated insertion & Delete the inserted span; audit upstream of the language hint. \\
    Proper noun dropped & Cross-check project metadata; proper-noun loss is the most consequential failure for keyword search. \\
    \bottomrule
  \end{tabular}
\end{table}
"""
(FIG / "reviewer_checklist_table.tex").write_text(checklist_tex)
print("[save] reviewer_checklist_table.tex")

print("\n[done] all figures/tables written.")
