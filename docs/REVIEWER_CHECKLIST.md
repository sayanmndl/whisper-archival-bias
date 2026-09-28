# Reviewer checklist for Whisper transcripts of South Asian diasporic audio

This is the full version of the reviewer checklist summarised in the paper
(Section V). It turns the word-level error typology into concrete actions for a
speaker of the language reviewing a machine transcript before it is promoted to
a searchable index.

The typology itself is computed by `experiments/06_make_figures_full.py`
(`categorise()`), on the *Ours* normalisation. The per-language counts are in
`results/error_type_counts.csv`.

## Before reviewing

1. **Treat the transcript as a draft.** Mark any Whisper transcript of South
   Asian audio as draft until reviewed, and make that status visible in the
   search interface. Fix a WER threshold for the collection rather than per
   language: an English transcript that crosses it needs the same treatment.
2. **Check the language hint.** Record which language hint (if any) the
   transcript was produced with. If the hint was English on audio that is
   mostly in another language, discard the transcript and re-run: in our case
   study, a forced-English hint on Bengali speech produced fluent English that
   mixed accurate translation with hallucinated content at LID confidence 1.0,
   and nothing in the output marks which is which.
3. **Check the detected language probability.** On FLEURS read speech the
   first-pass LID probability is near 1.0 for every language. On the
   spontaneous case-study clip it was 0.85. A noticeably lower value is a cue
   that the audio is outside the model's comfort zone.

## Per error mechanism

Ordered by consequence for retrieval. The first four are the ones in the
paper's summary table.

| Error mechanism | What it looks like | Reviewer action |
|---|---|---|
| Dropped non-English token | A word in the audio (Devanagari, Bengali, Tamil or Urdu script in the reference) is missing from the transcript. | Re-listen; restore the dropped term in the original script; tag it with the language code. A dropped word is unsearchable, so this is the highest-priority fix. |
| Script-to-anglicisation | A non-Latin word is replaced by a Latin-script token (e.g. Bengali গিগাহার্জ rendered as `ghz`, Urdu اس as `53`). | Verify the anglicised form against the speaker's preferred romanisation; record both the original-script and the romanised forms so either is findable. |
| Hallucinated insertion | Words appear in the transcript with no corresponding speech. | Delete the inserted span. If insertions cluster, check the language hint and decoding settings upstream before reviewing further. |
| Proper noun dropped or altered | A personal name, family name, place or village name is missing, anglicised or respelled. | Cross-check against project metadata (interview sheet, consent form, catalogue record). Proper-noun loss is the most consequential failure for keyword search: when a surname is lost, the story about that family is lost from the index. |
| Character substitution | A same-script word with different letters (the most frequent category in every language, 50% to 66% of error events). | Correct the spelling. These still leave a searchable string, so they are lower priority than drops, but they break exact-match search on the correct term. |
| Matra or word-form error | Same letters, different surface form: usually a missed or wrong vowel sign in Devanagari or Bengali, or a different inflection in Tamil or Urdu (34% of Hindi error events). | Correct the vowel sign or inflection. Consider whether the archive's search normalises these forms; if it does not, these errors hide the word from exact search. |
| Deletion (Latin-script) | An English or other Latin-script word is missing. Rare outside English. | Re-listen and restore, as for a dropped non-English token. |

## After reviewing

- Publish the raw machine transcript and the reviewed transcript side by
  side, with notes on the discrepancies, so later users can see what the
  model changed.
- Keep a record of which reviewer (or community member) reviewed which
  recording and in which language, so the reviewed status can be audited.
- Where the narrator is reachable, prefer the narrator's own correction of
  names and places over any external source.

## Scope

The typology is deterministic and was built on FLEURS read speech, not on
diasporic register. It flags *where* to look; it cannot substitute for review by
a speaker of the language, and it will miss errors that preserve the surface
form but change the meaning.
