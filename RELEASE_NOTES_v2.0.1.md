# v2.0.1 - naming and prose cleanup

This patch release keeps the v2.0.0 data and numerical results unchanged. It
uses descriptive file names and removes draft-style wording from the paper
sources, plotting scripts, comments, and result notes.

## Changes

- Renamed the scale and shared-bias analyses to describe their statistical
  role directly.
- Replaced anthropomorphic and argumentative file names with
  `shared_bias_screen`, `score_limitations`, and `scale_bias_analysis`.
- Renamed the figure build script and figure style guide.
- Rewrote code comments and result notes in plain technical language.
- Updated all LaTeX references, figure labels, README commands, and output
  paths to the new names.
- Regenerated the affected PDF and PNG figures without changing their data.

## Verification

- The curated supplement contains 140 files.
- The manuscript source compiles to a 33-page PDF with no unresolved citation,
  reference, or overfull-box warnings.
- Python sources pass `compileall`.
- Direct searches of the release tree find no draft tool names or superseded
  file names.
- SHA-256 for `SIMPAT_reproducibility_supplement_v2.0.1.zip`:

  `c93f37a4659f3d425838d3da7c4bf55ff80b3a622657f6aeaae25476220fab95`

All versions remain citable through the concept DOI:
<https://doi.org/10.5281/zenodo.21005573>.
