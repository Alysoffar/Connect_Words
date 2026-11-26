# LinkWords data pipeline

This package contains offline jobs for vocabulary, embeddings, and lexical
relationships. Large source datasets and model weights are intentionally not
downloaded during setup.

## Setup and tests

Use Python 3.11:

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

Run commands from the repository root. Downloading is explicit and includes a
large multilingual Numberbatch file and the BGE-M3 model:

```powershell
python -m pipeline download --lang all
python -m pipeline build --lang all
```

For one language, replace `all` with `en`, `de`, `es`, `ar`, or `ja`. The build
can also be run in smaller resumable stages with `--step vocab`, `--step vectors`,
or `--step wordnet`. The vector encoder chooses the available GPU automatically;
pass `--device cpu` to force CPU use.

The pipeline uses Stopwords ISO and LDNOOBW lists downloaded into ignored
`data/filters/`. Every vocabulary candidate must appear in Open Multilingual
WordNet or the configured Hunspell/AyaSpell language dictionary. Initial-capital
candidates are additionally excluded for English and Spanish, but kept for
German, Arabic, and Japanese. Because frequency and dictionary lists do not
provide complete proper-name labels, a curated supplementary deny list can be placed at
`data/filters/proper_nouns/{lang}.txt`; the CLI warns when that file is absent.
The pipeline logs the removal count for each filtering rule per language. The
files and vector caches are resumable and are not committed.
The Japanese anchor-frequency lookup uses MeCab with IPADIC; both dependencies
are installed from `pipeline/requirements.txt`.

The result includes `data/vocab/{lang}.json`, a float16 vector matrix plus a
JSON word-to-row index under `data/vectors/`, and WordNet associations in
`data/linkwords.sqlite`. Each matrix row is a weighted concatenation of
L2-normalized 300-dimensional Numberbatch and 1024-dimensional BGE-M3 vectors.
Its cosine equals the configured weighted blend when both sources are present.
The vocabulary target is 40,000 words; BGE-M3 is run for up to 60,000 entries
(all entries for the supported vocabulary size).

Thresholds in `shared/languages.json` remain `null` placeholders until they are
calibrated against the generated vectors; this pipeline does not guess them.

## Threshold calibration and puzzles

After building vocabularies, vectors, and WordNet data, calibrate thresholds:

```powershell
python -m pipeline calibrate --lang all --samples 20000 --percentile 98.25
```

The percentile must be between 98 and 98.5. Calibration writes a histogram PNG
and statistics JSON per language under `data/calibration/`, prints a comparison
table, and records the selected percentile and threshold in
`shared/languages.json`.

Generate daily and practice puzzle collections after calibration:

```powershell
python -m pipeline puzzles --lang ar --daily 30 --practice 500
```

Anchor candidates are OMW-tagged nouns, verbs, and adjectives in a medium-to-high
wordfreq Zipf range, excluding the language blocklist. The generator creates
the thresholded top-five graph, rejects near-synonym anchors, and uses BFS to
require a 4-6 edge shortest route and at least three distinct routes. Results
are written to `data/puzzles/{lang}.json`; rejection reasons and attempted
anchor pairs are written to `data/puzzles/{lang}.report.json`. Daily entries
receive consecutive ISO dates beginning with the current date. Accepted anchor
words are not reused across either collection. If the requested counts cannot
be met, the command fails after saving the rejection report.

## Data sources and licenses

- [wordfreq](https://github.com/rspeer/wordfreq) supplies frequency-ranked
  vocabulary candidates. Its code is Apache-2.0; its bundled data includes
  CC-BY-SA 4.0 and other source-specific terms. Preserve its `NOTICE.md`
  attribution and review that notice before redistributing generated data.
- [ConceptNet Numberbatch](https://github.com/commonsense/conceptnet-numberbatch)
  vectors are CC-BY-SA 4.0; the project code is MIT.
- [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) is MIT-licensed. Download a
  pinned model revision explicitly before building embeddings.
- [Open Multilingual WordNet](https://omwn.org/) is an open aggregation, but
  component wordnets retain their own licenses and attribution requirements.
  The downloader also retrieves NLTK's extended OMW data for language coverage
  such as German. Review component terms before distributing the resulting
  SQLite database.
- Supplemental German and Spanish Hunspell headwords come from the pinned
  [wooorm/dictionaries](https://github.com/wooorm/dictionaries) revision; their
  source dictionary licenses are GPLv2/GPLv3 for German and
  GPLv3/LGPLv3/MPL-1.1 for Spanish. Arabic headwords come from the pinned
  [AyaSpell](https://github.com/linuxscout/ayaspell) revision, tri-licensed
  GPL-2.0-or-later/LGPL-2.1-or-later/MPL-1.1-or-later. Review those terms before
  redistributing generated vocabularies.
- [Stopwords ISO](https://github.com/stopwords-iso/stopwords-iso) is MIT.
  [LDNOOBW](https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words)
  profanity lists are CC-BY-4.0.

The configured cosine blend uses Numberbatch at `0.25` and BGE-M3 at `0.75`.
When one vector source is absent for a word, weights are renormalized over the
available source. The final score is the maximum of the blended cosine and the
WordNet boost.
