# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

`classification_v2/` is the newer iteration of the `classification/` prototype
described in the parent `../CLAUDE.md` (the SSC Open Call 2026 proposal). It
trains a supervised classifier to separate **HII / star-forming** regions from
**AGN** using optical/NIR emission-line ratios, with the training labels coming
from **Cloudy 17.00 photoionization model grids** rather than observed galaxies.

The parent directory's `CLAUDE.md` governs the *proposal text*; this file
governs only the *code and data* in `classification_v2/`.

## Layout

- `classificator_test.py` — the entire codebase. Defines `LineRatioAGNClassifier`
  plus standalone helpers `plot_bpt_nii(...)`, `prepare_full_line_features_hbeta_norm(...)`,
  and a `__main__` driver.
- `C17_POPSTAR_1myr.dat` — HII/star-forming grid (POPSTAR SED, Chabrier IMF,
  1 Myr). This is the **HII** class.
- `C17_AGN_alpha{08,10,12,14,16,18,20}_efrac02_CNfix.dat` — AGN grids, one per
  UV power-law slope alpha (0.8-2.0), 2% free-electron stop criterion. These are
  the **AGN** class.
- `.dat` format: 5 comment lines starting with `#`, then a whitespace/tab-separated
  header row (line 6) with columns `12+log(O/H)`, `log(N/O)`, `log(U)` followed by
  emission-line fluxes (`OII_3727`, `OIII_5007`, `NII_6584`, `SII_6717,31`, ...).
  Line fluxes are normalised to Hbeta = 100.

## Running

```bash
python3 classificator_test.py
```

The `__main__` driver globs the grids in the working directory itself
(`C17_POPSTAR*.dat` -> HII class, `C17_AGN*.dat` -> AGN class, concatenated),
reads them with `read_cloudy_grid()`, builds features, trains both an RF and a
GB model, and prints evaluation + permutation importance. It raises
`FileNotFoundError` if either grid set is missing - there is no longer a
dummy-data fallback.

Requires `numpy`, `pandas`, `scikit-learn`, `matplotlib`. No `pyproject.toml`,
`requirements.txt`, tests, or CI exist - install deps manually.

The driver pops up matplotlib windows (`plt.show()`); on a headless box set
`MPLBACKEND=Agg` or the run will block/fail.

## Architecture

`LineRatioAGNClassifier` wraps an sklearn `Pipeline` of
`SimpleImputer(strategy="median")` -> `RandomForestClassifier` (`model_type="rf"`)
or `HistGradientBoostingClassifier` (`model_type="gb"`), optionally wrapped in
`CalibratedClassifierCV` (isotonic, cv=3, on by default). Class encoding is
fixed: **HII = 0, AGN = 1**.

Data flow: two same-columned DataFrames (`df_hii`, `df_agn`) -> `fit()` stacks
them, builds labels, stratified train/test split, fits, and on `verbose` prints
confusion matrix + classification report + ROC AUC. Inference: `predict_proba`,
`predict(threshold=0.5)`, and `classify(threshold_agn, threshold_hii)` which
adds a third "Composite/Uncertain" band between the two thresholds.
Diagnostics: `plot_roc_curve()` and `plot_permutation_importance()` (importance =
drop in ROC AUC when a feature is shuffled).

`prepare_full_line_features_hbeta_norm()` turns a raw line-flux table into
`log10(line / Hbeta)` features, summing doublets (SII, OIII, OII) where present.

## Important gotchas

- **Grid fluxes are normalised to Hbeta = 1, not 100.** The driver passes
  `fluxHb=1.0` to `prepare_full_line_features_hbeta_norm`. (A wrong scale is only
  a constant per-feature offset shared by both classes, so it would not change
  the tree-based models, but it would make the log ratios physically wrong.)
- **`line_list` must match the `.dat` headers exactly**, including the doublet
  column `SII_6717,31` (comma included) and `NII_6584` (not `6583`). Any name not
  present is filled with NaN by `prepare_full_line_features_hbeta_norm` and then
  dropped by the >50%-NaN filter, so a typo silently removes a feature.
- **Separation is essentially perfect (ROC AUC ~ 1.0).** On these grids the
  classes are cleanly separable, driven almost entirely by `HeII_4686` (strong in
  AGN, absent in HII); permutation importance of every other line is ~0. This is
  expected for noise-free photoionization models and is *not* representative of
  real, noisy observations - treat the AUC as a sanity check, not a performance
  claim.
- Built with LLM assistance; the user wants ML-expert review (calibration and
  validation on a grid-derived, not observation-derived, training set is a known
  soft spot).

## Version control

Not a git repository (consistent with the parent directory).
