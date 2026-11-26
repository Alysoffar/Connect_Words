# Shared configuration

`languages.json` is the canonical source for per-language word-length limits,
cosine-similarity threshold, right-to-left layout flag, and font family.
It also configures whether vocabulary building excludes candidates that begin
with a capital letter; this heuristic is enabled only for English and Spanish.

The offline vocabulary builder additionally requires each normalized candidate
to occur in the corresponding Open Multilingual WordNet lemma list. An optional
`data/filters/proper_nouns/{lang}.txt` deny list is applied as an extra filter.

Thresholds are `null` placeholders until they are calibrated for the selected
embedding model and vocabulary. Consumers must treat an unset threshold as
unconfigured rather than silently accepting or rejecting links.

Word lengths are initial configuration values; their normalization and counting
semantics belong in the shared normalization implementation.
