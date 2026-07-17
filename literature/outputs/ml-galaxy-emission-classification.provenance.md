# Provenance: ML contributions to galaxy emission-line classification

- **Date:** 2026-07-13
- **Rounds:** 1 planned research round with 4 researcher briefs; 1 citation verification pass; 1 reviewer-style verification pass; 1 revision pass.
- **Sources consulted:** 44 BibTeX entries, 38 downloaded PDFs, 60 URL checks, plus official survey/data documentation.
- **Sources accepted:** Classical diagnostics (Baldwin 1981; Veilleux & Osterbrock 1987; Kewley 2001/2006; Kauffmann 2003; Cid Fernandes 2011), ML papers (Zhang 2019; Teimoorinia & Keown 2018; Ding & Rodriguez 2024; de Souza 2017; Rhea 2023; Pat 2023; Wang 2023; Gupta 2024; Cho 2026; Fraix-Burnet 2021; Teimoorinia/DESOM 2022; Melchior/SPENDER 2022; Liang/SPENDER 2023; Alcolea/DESI autoencoder AGN 2026; Parker/AstroCLIP 2024; Pattnaik/SpecPT 2025; Khederlarian emission-line prediction 2024), data/validation sources (SDSS/MPA-JHU, MaNGA DAP, CALIFA DR3, SAMI DR3, Belfiore 2016/2019, Kewley 2005, Richards 2015), photoionization/simulation sources (Ferland 2017, Levesque 2010, Allen 2008, Zhu 2023, Meléndez 2014), and calibration/domain-shift sources (Shy 2022, Guo 2017, Humphrey 2022).
- **Sources rejected or downgraded:** Local project code/data and the earlier scratch local-run log were not used as evidence. Publisher PDFs that could not be downloaded are listed in `outputs/.sources/ml-galaxy-emission-classification/pdf-manifest.md`; claims from those papers were kept summary-level and marked by source-type caveats. X-ray/UV AGN-validation claims were downgraded because no dedicated X-ray/UV selection review was performed.
- **Verification:** PASS WITH NOTES. Reviewer MAJOR findings W1-W4 were addressed by narrowing tree/ensemble claims, softening synthetic-grid domain-shift language, removing strong X-ray/UV validation claims, and adding source-type caveats. URL checks passed for arXiv and official documentation pages; several publisher DOI pages resolved but returned 403/security pages, recorded in `outputs/.sources/ml-galaxy-emission-classification/url-checks.md`.
- **Plan:** `outputs/.plans/ml-galaxy-emission-classification.md`
- **Research files:**
  - `outputs/.drafts/ml-galaxy-emission-classification-research-diagnostics.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-research-ml.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-research-data.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-research-project.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-draft.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-cited.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-verification.md`
  - `outputs/.drafts/ml-galaxy-emission-classification-revised.md`
  - `outputs/.sources/ml-galaxy-emission-classification/pdf-manifest.md`
  - `outputs/.sources/ml-galaxy-emission-classification/url-checks.md`
- **Final report:** `outputs/ml-galaxy-emission-classification.md`
- **BibTeX:** `outputs/ml-galaxy-emission-classification.bib`
- **PDF directory:** `outputs/.sources/ml-galaxy-emission-classification/pdfs/`
- **Blocked / notes:** Full PDF download was blocked or unavailable for several publisher-hosted papers; this is recorded in the PDF manifest. No further project code/data inspection was performed after user clarification, and local code/data are not used as sources.

## Fix verification commands

The following on-disk checks were run after revision:

- Removed/softened unsupported phrases: `rg -n "main unresolved issue|strongest practical message|mid-IR, X-ray, radio, UV|WISE/X-ray/radio|likely publishable|research-brief evidence|often provide a better interpretability" outputs/.drafts/ml-galaxy-emission-classification-revised.md` returned no matches.
- Added corrected wording: `rg -n "strong baselines|ambiguous or physically mixed|design risk|X-ray/UV validation remains|Source-type caveat|Revision note" outputs/.drafts/ml-galaxy-emission-classification-revised.md` returned matches.
- Final artifact copied and checked with `stat outputs/ml-galaxy-emission-classification.md`.

## Update: semi/self-supervised and generative section

At the user's request, the final report was extended with `## 4.5 Semi-supervised, self-supervised, unsupervised, and generative directions`. Added sources include unsupervised SDSS spectral clustering, MaNGA DESOM maps, SPENDER autoencoding/outlier detection, DESI semi-supervised AGN discovery with autoencoder embeddings, AstroCLIP, SpecPT, and differentiable emission-line prediction. Eight additional arXiv PDFs were downloaded and eight BibTeX entries were added. Added URLs returned HTTP 200 in `outputs/.sources/ml-galaxy-emission-classification/url-checks.md`.
