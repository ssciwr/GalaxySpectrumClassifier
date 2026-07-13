# Machine Learning Contributions to Galaxy Emission-Line Classification

## Executive summary

The scientific problem is not merely “classify a spectrum”; it is to infer the dominant ionization source behind nebular emission. Classical line-ratio diagnostics, especially the BPT/VO87 family, separate HII-region/star-formation-like emission from harder ionization fields associated with AGN, but they do so with known gray zones: composites, low-ionization emission regions, shocks, diffuse ionized gas, weak lines, aperture effects, and metallicity/N/O degeneracies [@Baldwin1981BPT; @Veilleux1987VO; @Kewley2006AGNClassification; @CidFernandes2011WHAN; @Kewley2019EmissionLines].

Machine learning has contributed in four main ways:

1. **Operational classifiers for survey data.** Supervised methods such as random forests, SVMs, MLPs, dense neural networks, and related ensemble models can reproduce or extend BPT-style labels using compact line-ratio features, colors, kinematics, or multi-wavelength data. Zhang et al. and Ding & Rodriguez support random forests as strong baselines in their respective feature/label settings; Gupta et al. shows that dense neural networks and SVMs are also used for the same taxonomy. The safer practical message is therefore baseline discipline: compare any new model against simple, physically interpretable feature sets and standard ML baselines before claiming architectural novelty [@Zhang2019MLCELG; @DingRodriguez2024Multiwavelength; @Gupta2024EmissionDNN].
2. **Classification when key BPT lines are unavailable.** Several studies target redshift regimes or instruments where Hα and [N II] are missing, using alternative features such as [O III], Hβ, equivalent widths, Dn4000, broadband colors, and line widths [@Teimoorinia2018NoHalphaNII; @Zhang2019MLCELG].
3. **Probabilistic and unsupervised boundary analysis.** Gaussian mixtures, UMAP/HDBSCAN, and related representation methods are useful where classical demarcation curves force hard labels on ambiguous or physically mixed regions [@DeSouza2017ProbELG; @Cho2026DangerZone; @Pat2023ReconstructingSDSS].
4. **Simulation-trained classifiers.** Neural networks and other models trained on photoionization/shock grids, including Cloudy-based model databases, show a plausible path for classifying emission-line regions from physically generated labels. A major design risk is domain shift from idealized grids to real observations. The cited literature establishes grid/model dependence and general astronomy population-shift risks; it does not, by itself, prove the size of the shift for every Cloudy-trained HII/AGN classifier [@Ferland2017Cloudy; @Allen2008ShockGrids; @Rhea2023EmissionRegionNN; @Zhu2023NLRPhotoionization; @Humphrey2022PopulationShift].
5. **Representation learning and generation.** Unsupervised clustering, autoencoders, contrastive/self-supervised encoders, transformer pretraining, and differentiable line-generation models broaden the task from hard classification to latent-space discovery, outlier detection, missing-line handling, and realistic mock spectra [@FraixBurnet2021UnsupervisedSDSS; @Melchior2022SpenderI; @Alcolea2026DESIAutoencoderAGN; @Parker2024AstroCLIP; @Pattnaik2025SpecPT; @Khederlarian2024EmissionLinePredictions].

For a project focused on HII versus AGN using Cloudy grids, the defensible contribution is not “we achieved high random-split accuracy on grids.” The stronger contribution would be a **reproducible validation framework**: physically grouped splits, cross-grid or cross-code tests, noise/upper-limit perturbations, uncertainty calibration, abstention for composites/OOD cases, and validation against observational survey labels without treating BPT labels as ground truth [@Ferland2017Cloudy; @Kewley2001MaxStarburst; @Zhu2023NLRPhotoionization; @Shy2022MeasurementError; @Guo2017Calibration; @Humphrey2022PopulationShift].

## 1. Problem framing: what is being classified?

Classical emission-line classification is based on the fact that different ionization mechanisms populate different regions of low-dimensional line-ratio space. Baldwin, Phillips & Terlevich introduced line-ratio diagrams as empirical classification parameters for extragalactic emission-line spectra [@Baldwin1981BPT]. Veilleux & Osterbrock formalized optical diagnostic diagrams using ratios such as [O III] λ5007/Hβ versus [N II] λ6584/Hα, [S II] λλ6717,6731/Hα, and [O I] λ6300/Hα [@Veilleux1987VO]. These ratios are close in wavelength within each pair, reducing sensitivity to reddening and flux-calibration errors [@Veilleux1987VO].

Kewley et al. used stellar-population synthesis plus MAPPINGS photoionization models to define a theoretical “maximum starburst” boundary [@Kewley2001MaxStarburst]. Kauffmann et al. used SDSS galaxies to define a more empirical division between pure star-forming systems and systems requiring an AGN-like component [@Kauffmann2003AGNHosts]. Kewley et al. later formalized a multi-diagram scheme that separates star-forming, composite, Seyfert, LINER, and ambiguous classes [@Kewley2006AGNClassification].

For machine learning, these classical diagnostics matter because they define both the feature space and many of the labels. But they are not pure ground truth. The “composite” region between empirical and theoretical curves is explicitly a mixed or ambiguous zone, and low-ionization emission can arise from AGN, shocks, diffuse gas, or old stellar populations rather than a single central accreting black hole [@Kewley2006AGNClassification; @Stasinska2008Retired; @CidFernandes2011WHAN; @Belfiore2016LIER; @Kewley2019EmissionLines].

## 2. Classical baselines that any ML contribution should beat or complement

A credible ML contribution should compare against at least these baselines:

- **[N II]-BPT:** log([O III] λ5007/Hβ) versus log([N II] λ6584/Hα), with Kauffmann and Kewley demarcation curves [@Kewley2001MaxStarburst; @Kauffmann2003AGNHosts].
- **[S II]-BPT and [O I]-BPT:** useful for separating Seyfert and LINER-like branches and for sensitivity to shocks/low-ionization emission [@Veilleux1987VO; @Kewley2006AGNClassification; @Kewley2019EmissionLines].
- **WHAN:** [N II]/Hα plus Hα equivalent width, designed to distinguish weak AGN from “retired” galaxies whose line emission can be powered by old stellar populations [@CidFernandes2011WHAN; @Stasinska2008Retired].
- **Spatially resolved variants:** IFU surveys are essential because nuclear, aperture-integrated, and spaxel-level views can differ. The strongest cited evidence here is from MaNGA LIER/emission-line work and SAMI aperture-correction work; CALIFA is included as a relevant IFU data resource, not as separately verified evidence for this specific claim [@Belfiore2016LIER; @Belfiore2019MaNGADAP; @Richards2015SAMIaperture].

Important caveat: BPT/WHAN labels are often used as ML targets, so high ML accuracy may simply mean “the model learned the demarcation rule,” not that it has discovered a better physical source classifier. This is acceptable if the task is label propagation, but it is weaker if the scientific claim is ionization-source inference [@Kewley2006AGNClassification; @CidFernandes2011WHAN; @Zhang2019MLCELG; @Gupta2024EmissionDNN].

## 3. Supervised ML on emission-line and survey features

### 3.1 Line-ratio and compact-feature classifiers

Several studies apply standard supervised models to emission-line galaxies. Zhang et al. compared KNN, SVC, random forests, and MLPs for intermediate-redshift emission-line galaxy classification, using features such as [O III]/Hβ, [O II]/Hβ, [O III] width, stellar velocity dispersion, and ugriz colors [@Zhang2019MLCELG]. Their public repository makes this a useful reproducibility reference [Zhang et al. repository: https://github.com/zkdtc/MLC_ELGs].

Teimoorinia & Keown addressed classification when Hα and [N II] are unavailable, using ANN pattern recognition on SDSS-derived features [@Teimoorinia2018NoHalphaNII]. The key contribution is methodological: when a canonical BPT axis is inaccessible, one can learn a proxy classifier from other spectral features and continuum indicators, but this inherits the assumptions and selection effects of the training labels [@Teimoorinia2018NoHalphaNII; @Kauffmann2003AGNHosts].

Wang et al. trained MLP/SVM/KNN/RF-like models directly on spectral flux windows around emission lines for LAMOST emission-line galaxies [@Wang2023LAMOSTELG]. This approach reduces manual ratio engineering, but it still depends on how the spectral windows are normalized, how labels are defined, and whether line S/N and continuum subtraction errors are propagated [@Wang2023LAMOSTELG; @Shy2022MeasurementError; @Belfiore2019MaNGADAP].

Gupta et al. used dense neural networks and SVMs on SDSS/BOSS emission-line ratios for star-forming, Seyfert, LINER, and composite classes [@Gupta2024EmissionDNN]. This is useful as a recent example of neural-network methods applied to the standard taxonomy, but it should be interpreted as label reproduction unless validated against independent AGN indicators [@Gupta2024EmissionDNN; @Kewley2006AGNClassification].

### 3.2 Multi-wavelength supervised classification

Ding & Rodriguez studied multi-wavelength supervised classification of active and star-forming galaxies on the BPT diagram, crossmatching SDSS DR16 with WISE, GALEX, and XMM information [@DingRodriguez2024Multiwavelength]. Their reported random-forest results show that combining photometry and spectroscopy can improve classification, while IR-only classification may have weaker AGN sensitivity [@DingRodriguez2024Multiwavelength]. The broader lesson is that optical line ratios alone are not always enough: mid-IR and radio selections trace AGN populations with different selection functions, and Ding & Rodriguez demonstrate one supervised multi-wavelength workflow using SDSS, WISE, GALEX, and XMM crossmatches. This report does not separately audit X-ray or UV AGN-selection literature, so those bands should be treated as candidate validation channels needing dedicated references rather than as fully reviewed evidence here [@Stern2012WISEAGN; @BestHeckman2012RadioAGN; @DingRodriguez2024Multiwavelength].

For a HII-versus-AGN project, multi-wavelength labels are valuable not necessarily as input features, but as **external validation**. Disagreements between an optical classifier and mid-IR/radio AGN selections should be analyzed rather than treated as simple errors; X-ray or UV validation should be added only with dedicated source support [@Stern2012WISEAGN; @BestHeckman2012RadioAGN; @DingRodriguez2024Multiwavelength].

## 4. Probabilistic, unsupervised, and representation-learning contributions

A hard star-forming/AGN boundary is often physically unrealistic. de Souza et al. proposed a probabilistic approach to emission-line galaxy classification using Gaussian mixture models in a combined BPT/WHAN-like space [@DeSouza2017ProbELG]. The value is not only classification, but calibrated membership and recognition that the data structure may not match the historical hand-drawn classes [@DeSouza2017ProbELG; @CidFernandes2011WHAN].

Cho et al. used UMAP and HDBSCAN on MaNGA spaxel line-ratio features to analyze ambiguous boundary regions and define buffer zones near classical BPT demarcations [@Cho2026DangerZone]. Whether or not one adopts their exact buffer definitions, the contribution is conceptually important: the decision boundary itself should be modeled as uncertain [@Cho2026DangerZone; @Kewley2006AGNClassification].

Pat et al. used probabilistic autoencoders and UMAP on full SDSS spectra to reconstruct and classify galaxy spectra and identify transitions/outliers [@Pat2023ReconstructingSDSS]. Full-spectrum representation learning is promising for discovery and anomaly detection, but it should not be treated as a black-box replacement for line-ratio physics without interpretability and source-support checks [@Pat2023ReconstructingSDSS; @Baldwin1981BPT; @Veilleux1987VO].

## 4.5 Semi-supervised, self-supervised, unsupervised, and generative directions

The supervised papers above are only one part of the relevant ML landscape. For emission-line source classification, representation learning can be useful even when the final scientific question is still “HII-like or AGN-like?” because the hard part is often not the classifier head; it is building a spectral representation that preserves weak emission lines, continuum shape, redshift effects, noise properties, and rare/outlier populations.

### Unsupervised clustering and maps

Unsupervised clustering can expose structure before committing to BPT/WHAN labels. Fraix-Burnet et al. applied Fisher-EM, a discriminative latent mixture model, to 702,248 SDSS DR7 galaxy/quasar spectra after redshift correction, wavelet filtering, and spectral binning; the optimum solution contained 86 classes, with the 37 most populated classes covering 99% of the sample [@FraixBurnet2021UnsupervisedSDSS]. Teimoorinia et al. used a deep embedded self-organizing map on MaNGA spectra to summarize spectral diversity onto a 15 × 15 DESOM map, then represented each galaxy by its distribution over spectral cells, a “fingerprint” connected to stellar populations, star-formation histories, morphology, and inclination [@Teimoorinia2022DESOM].

For HII/AGN work, these methods are not replacements for physical diagnostics. Their value is exploratory: they can reveal whether spectra labeled “composite,” “retired,” or “AGN” actually form coherent groups; whether ambiguous regions are continuous transitions or separate populations; and whether outliers deserve new labels rather than forced binary classification.

### Autoencoders and latent spaces for full spectra

Autoencoders are especially relevant because they can learn from the full spectrum while still producing low-dimensional features usable by downstream classifiers. SPENDER is a galaxy-spectrum autoencoder that combines a convolutional encoder with attention over spectral features and a decoder that generates rest-frame spectra before applying explicit redshift, resampling, and instrumental transformations [@Melchior2022SpenderI]. In the follow-up outlier-detection paper, the authors used a normalizing flow over the SPENDER latent space to identify low-probability spectra, finding blends with foreground stars, reddened galaxies, galaxy pairs/triples, and misclassified stars among the outliers [@Liang2023SpenderII].

This suggests a concrete semi-supervised design for ionization-source classification: train or reuse a spectral autoencoder without HII/AGN labels, inspect the latent space for BPT/WHAN/cross-wavelength structure, and then train a small supervised classifier or nearest-neighbor rule on the subset with reliable labels. The representation can also support abstention: sources far from labeled regions, or with high reconstruction error / low latent density, can be flagged as “unknown/OOD” rather than forced into HII or AGN.

### Semi-supervised AGN discovery in DESI

A direct recent example is Alcolea et al. (2026), which uses SPENDER representations of DESI spectra inside a semi-supervised classification framework for AGN discovery [@Alcolea2026DESIAutoencoderAGN]. The method compresses 50,222 DESI Main Survey spectra into a low-dimensional latent space and classifies sources with a k-d-tree nearest-neighbor search using FastSpecFit-derived labels for seven classes: AGN, broad-line AGN, composite, star-forming, passive, retired, and Other [@Alcolea2026DESIAutoencoderAGN]. The abstract reports AGN and broad-line AGN accuracies of 0.952 and 0.965, respectively, and emphasizes recovery of AGN missed by single diagnostic methods; because this is a very recent preprint, it should be treated as promising evidence rather than a settled benchmark [@Alcolea2026DESIAutoencoderAGN].

This is close to the user’s problem space because it changes the workflow from “classify a few line ratios” to “learn the full spectral manifold, then label or query it.” It is particularly relevant for noisy spectra, missing-line regimes, and composite/retired classes, where line-ratio-only decision boundaries are known to be fragile.

### Self-supervised and foundation-style spectra models

Self-supervised pretraining is beginning to appear for galaxy spectra. AstroCLIP pretrains image and spectrum encoders, then aligns them with a contrastive loss so that galaxy images and DESI spectra occupy a shared latent space [@Parker2024AstroCLIP]. Its reported downstream tasks are property estimation, photometric redshifts, similarity search, and morphology rather than HII/AGN classification, but the approach is relevant because it provides a label-efficient representation that can be queried or fine-tuned for emission-line source problems [@Parker2024AstroCLIP]. SpecPT is a transformer model for DESI spectroscopy trained for spectrum reconstruction and redshift measurement; its abstract reports that it reconstructs spectra while capturing emission lines, absorption features, and continua, and it proposes the learned latent representation as a basis for outlier detection, ISM-property estimation, and transfer learning [@Pattnaik2025SpecPT].

The contribution opportunity is to use these models as frozen or lightly fine-tuned encoders and ask whether their latent spaces separate BPT/WHAN classes, cross-wavelength AGN candidates, shocks, and retired/LIER systems better than hand-built line ratios alone. A fair comparison would keep the same labels and validation splits across: line-ratio baselines, full-spectrum autoencoder embeddings, contrastive/self-supervised embeddings, and supervised full-spectrum models.

### Generative and differentiable emission-line modeling

Generative modeling is not only for producing pretty spectra. It can support uncertainty, simulation, and missing-data handling. SPENDER-like decoders generate rest-frame spectra conditioned on latent variables and observational transformations [@Melchior2022SpenderI]. Liang et al. add a normalizing flow over latent space to estimate sample density and detect anomalous spectra [@Liang2023SpenderII]. Khederlarian et al. train a differentiable neural network on DESI Early Release data to predict equivalent widths for eight bright optical emission lines, including Hα, Hβ, [O II], and [O III], from rest-frame optical continua; they report Spearman correlations above 0.87 for most lines and show that adding measurement uncertainties is essential to reproduce observed BPT line-ratio distributions [@Khederlarian2024EmissionLinePredictions].

For HII/AGN classification, these models suggest several extensions: generate realistic line-strength/noise realizations for robustness tests; impute missing lines with calibrated uncertainty instead of single point estimates; create mock catalogs whose BPT/WHAN distributions match survey data; and test whether a classifier’s decision boundary is stable under plausible continuum-to-line mappings. The caveat is that generation can easily hide assumptions: a generative model trained on observed survey labels or selection functions will reproduce those biases unless explicitly corrected.

### Practical implications

Semi-supervised and unsupervised approaches are most useful when labels are scarce, noisy, or conceptually incomplete. In this problem space, they should be framed as tools for representation, discovery, uncertainty, and data curation, not as automatic replacements for physical diagnostics. A strong workflow would combine: (1) classical BPT/WHAN labels as anchors; (2) autoencoder or self-supervised spectral embeddings; (3) density/outlier scores for abstention; (4) small supervised heads or nearest-neighbor label propagation; and (5) external checks from IFU structure and multi-wavelength AGN indicators.

## 5. Simulation-trained and photoionization-grid ML

Photoionization grids are attractive because they provide physically controlled labels and parameter coverage. Cloudy and MAPPINGS are widely used to generate nebular line predictions for HII regions, AGN narrow-line regions, shocks, and related sources [@Ferland2017Cloudy; @Kewley2001MaxStarburst; @Levesque2010Starforming; @Allen2008ShockGrids; @Zhu2023NLRPhotoionization].

Rhea et al. is particularly relevant to simulation-trained emission-region classification: it uses the Million Mexican Model database, including Cloudy photoionization models and shock models, to train a neural network on diagnostic line ratios for HII regions, planetary nebulae, and supernova remnants [@Rhea2023EmissionRegionNN]. This is not an HII-versus-AGN classifier, but it is direct evidence that grid-trained ML for emission-line region classification is an active and plausible methodology [@Rhea2023EmissionRegionNN].

For AGN/HII separation, Kewley et al., Levesque et al., Allen et al., Meléndez et al., and Zhu et al. collectively show why the model grid matters [@Kewley2001MaxStarburst; @Kewley2006AGNClassification; @Levesque2010Starforming; @Allen2008ShockGrids; @Melendez2014AGNDiagnostics; @Zhu2023NLRPhotoionization]. Predicted line ratios depend on stellar population assumptions, gas density, metallicity, N/O, ionization parameter, dust, AGN spectral energy distribution, and the chosen code [@Kewley2001MaxStarburst; @Levesque2010Starforming; @Zhu2023NLRPhotoionization]. Therefore, a classifier trained on one synthetic grid family should be treated as potentially brittle outside that grid family until cross-grid or observational transfer tests show otherwise [@Zhu2023NLRPhotoionization; @Humphrey2022PopulationShift].

The research opportunity is to treat synthetic grids as **controlled training environments**, not as a substitute for observational validation. A strong study design would ask: which distinctions are robust across grid assumptions, and which are artifacts of a particular grid design? [@Zhu2023NLRPhotoionization; @Shy2022MeasurementError; @Humphrey2022PopulationShift]

## 6. Datasets and labels

Common observational sources include SDSS/MPA-JHU products, MaNGA DAP maps, CALIFA datacubes, SAMI DR3 products, OSSY line measurements, and multi-wavelength catalogs such as WISE mid-infrared and radio AGN samples [SDSS MPA-JHU documentation: https://www.sdss3.org/dr9/algorithms/galaxy_mpa_jhu.php; MaNGA DAP documentation: https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html; CALIFA DR3: https://califa.caha.es/CALIFA_3rd_DATA_RELEASE.html; SAMI DR3: https://sami-survey.org/node/902; @Oh2011OSSY; @Stern2012WISEAGN; @BestHeckman2012RadioAGN].

Label construction is the central weakness of many ML papers in this area because BPT, WHAN, cross-wavelength, and simulation labels each encode different selection rules rather than a single ground-truth oracle [@Kewley2006AGNClassification; @CidFernandes2011WHAN; @Stern2012WISEAGN; @Zhu2023NLRPhotoionization]. Common label sources include:

- BPT class labels from SDSS-style line ratios [@Baldwin1981BPT; @Veilleux1987VO; @Kauffmann2003AGNHosts; @Kewley2006AGNClassification].
- WHAN labels using Hα equivalent width to separate weak AGN from retired galaxies [@CidFernandes2011WHAN; @Stasinska2008Retired].
- Cross-wavelength AGN indicators such as WISE mid-IR color cuts or radio AGN catalogs; Ding & Rodriguez also use GALEX and XMM crossmatches, but this report does not independently review UV/X-ray AGN-selection criteria [@Stern2012WISEAGN; @BestHeckman2012RadioAGN; @DingRodriguez2024Multiwavelength].
- Simulation labels from photoionization/shock grids [@Ferland2017Cloudy; @Allen2008ShockGrids; @Rhea2023EmissionRegionNN; @Zhu2023NLRPhotoionization].

None of these is a perfect oracle. BPT labels are diagnostic categories, not direct physical truth. Cross-wavelength AGN indicators select different AGN populations and have their own selection functions. Simulation labels are only as good as the parameter coverage and physics in the grid [@Kewley2006AGNClassification; @CidFernandes2011WHAN; @Stern2012WISEAGN; @BestHeckman2012RadioAGN; @Zhu2023NLRPhotoionization].

## 7. Failure modes and caveats

### 7.1 Composite regions and mixed ionization

Composite regions can contain both star formation and AGN-like ionization, and an integrated spectrum may mix physically distinct regions. A binary classifier that forces every object into HII or AGN will overstate certainty in exactly the region where classical diagnostics are most cautious [@Kauffmann2003AGNHosts; @Kewley2001MaxStarburst; @Kewley2006AGNClassification; @DeSouza2017ProbELG; @Cho2026DangerZone].

### 7.2 LINER/LIER and retired galaxies

Stasińska et al., Cid Fernandes et al., and MaNGA work by Belfiore et al. show that low-ionization emission can be powered by old stellar populations or extended low-ionization regions, not only AGN [@Stasinska2008Retired; @CidFernandes2011WHAN; @Belfiore2016LIER]. WHAN-style equivalent-width information and spatial context are important safeguards [@CidFernandes2011WHAN; @Belfiore2016LIER; @Belfiore2019MaNGADAP].

### 7.3 Shocks and diffuse ionized gas

Shock models and IFU diagnostics show that shocks can overlap with AGN-like line-ratio space [@Allen2008ShockGrids; @Kewley2019EmissionLines]. D’Agostino et al. proposed a 3D diagnostic using line ratios, velocity dispersion, and radius to separate star formation, shocks, and AGN in IFU data [@DAgostino2019IFUDiagnostic]. This supports a broader ML design principle: when shocks are plausible, line ratios alone may be insufficient [@DAgostino2019IFUDiagnostic; @Kewley2019EmissionLines].

### 7.4 Aperture effects

Single-fiber spectra can mix nuclear and disk emission differently depending on redshift, galaxy size, and aperture coverage. Kewley et al. and Richards et al. show that aperture effects affect star-formation, metallicity, reddening, and aperture-corrected inferences [@Kewley2005Aperture; @Richards2015SAMIaperture]. Spatially resolved IFU data mitigate but do not eliminate this issue [@Belfiore2016LIER; @Belfiore2019MaNGADAP].

### 7.5 Measurement error, censoring, and missing lines

Astronomical emission-line data are heteroscedastic: each line has its own flux uncertainty, upper limits, and detection thresholds. Shy et al. discusses measurement-error propagation in astronomical object classification [@Shy2022MeasurementError]. A robust ML classifier should report sensitivity to line S/N, upper limits, and missing-line masks rather than only point-estimate accuracy [@Shy2022MeasurementError; @Belfiore2019MaNGADAP].

### 7.6 Calibration and population shift

Probability calibration matters if a classifier output is used as P(AGN). scikit-learn’s calibration documentation and Guo et al. both emphasize that classifier scores are not automatically calibrated probabilities [scikit-learn User Guide: https://scikit-learn.org/stable/modules/calibration.html; @Guo2017Calibration]. Humphrey et al. shows that population shift can degrade astronomical classifiers without immediate access to new labels [@Humphrey2022PopulationShift]. For synthetic-grid classifiers, population shift should be treated as a default design risk when applying grids to survey data; the magnitude of that risk must be measured rather than assumed [@Zhu2023NLRPhotoionization; @Humphrey2022PopulationShift].

## 8. What counts as a real ML contribution here?

Weak contributions:

- Training a generic classifier to reproduce BPT labels without comparing against the BPT rules [@Kauffmann2003AGNHosts; @Kewley2001MaxStarburst; @Gupta2024EmissionDNN].
- Reporting high random train/test accuracy on synthetic grids without grouped or out-of-distribution tests; the literature supports this as a domain-shift risk, not as a quantified failure rate for every grid classifier [@Zhu2023NLRPhotoionization; @Humphrey2022PopulationShift].
- Treating composite, LINER/LIER, shock, and weak-line regions as simple binary errors [@Kewley2006AGNClassification; @CidFernandes2011WHAN; @Belfiore2016LIER; @Allen2008ShockGrids].
- Reporting classifier probabilities without calibration checks [scikit-learn User Guide: https://scikit-learn.org/stable/modules/calibration.html; @Guo2017Calibration].

Stronger contributions:

1. **Benchmark classical diagnostics explicitly.** Compare against Kauffmann/Kewley/WHAN rules and report where ML differs [@Kauffmann2003AGNHosts; @Kewley2001MaxStarburst; @CidFernandes2011WHAN].
2. **Use physically grouped validation.** Hold out metallicity, N/O, ionization parameter, AGN continuum slope, density, or full model-family subsets when training on grids [@Kewley2001MaxStarburst; @Levesque2010Starforming; @Zhu2023NLRPhotoionization].
3. **Add observation-like perturbations.** Inject realistic flux noise, upper limits, reddening uncertainty, continuum-subtraction uncertainty, and missing-line patterns [@Shy2022MeasurementError; @Belfiore2019MaNGADAP].
4. **Quantify uncertainty and abstention.** Use calibrated probabilities, conformal/abstention bands, or explicit composite/unknown classes [@DeSouza2017ProbELG; @Cho2026DangerZone; @Guo2017Calibration].
5. **Validate against external evidence.** Use SDSS/MaNGA survey labels, IFU resources such as CALIFA/SAMI where appropriate, and mid-IR/radio AGN catalogs as imperfect but informative checks [SDSS MPA-JHU documentation: https://www.sdss3.org/dr9/algorithms/galaxy_mpa_jhu.php; MaNGA DAP documentation: https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html; CALIFA DR3: https://califa.caha.es/CALIFA_3rd_DATA_RELEASE.html; SAMI DR3: https://sami-survey.org/node/902; @Stern2012WISEAGN; @BestHeckman2012RadioAGN].
6. **Probe shortcut features.** Use feature ablations to determine whether the classifier relies on one line or a physically robust combination [@Zhang2019MLCELG; @DingRodriguez2024Multiwavelength; @Zhu2023NLRPhotoionization].
7. **Include contaminants.** Test shock, planetary-nebula, retired-galaxy, and diffuse-ionized-gas models as out-of-distribution or third-class examples [@Allen2008ShockGrids; @Rhea2023EmissionRegionNN; @Stasinska2008Retired; @Belfiore2016LIER; @Kewley2019EmissionLines].

## 9. Recommended research direction for a Cloudy-grid HII-versus-AGN classifier

A good project framing would be [@Ferland2017Cloudy; @Zhu2023NLRPhotoionization; @Shy2022MeasurementError; @Humphrey2022PopulationShift]:

> We train classifiers on controlled photoionization grids to distinguish HII-like and AGN-like ionization, then evaluate whether the learned boundary is robust to physical parameter holdouts, observational noise, missing lines, calibration, and known contaminant classes [@Ferland2017Cloudy; @Zhu2023NLRPhotoionization; @Shy2022MeasurementError; @Guo2017Calibration; @Humphrey2022PopulationShift].

This framing is supported by the use of Cloudy/MAPPINGS-style grids in emission-line modeling, the documented model-dependence of AGN/HII predictions, and astronomy-specific risks from measurement error and population shift [@Ferland2017Cloudy; @Kewley2001MaxStarburst; @Allen2008ShockGrids; @Zhu2023NLRPhotoionization; @Shy2022MeasurementError; @Humphrey2022PopulationShift].

The most defensible evaluation plan:

1. **Baseline:** implement BPT/VO87/WHAN rules where the required lines exist [@Baldwin1981BPT; @Veilleux1987VO; @Kewley2001MaxStarburst; @Kauffmann2003AGNHosts; @CidFernandes2011WHAN].
2. **Grid-aware splits:** hold out full ranges or files corresponding to physical parameters rather than random rows only [@Levesque2010Starforming; @Zhu2023NLRPhotoionization; @Humphrey2022PopulationShift].
3. **Noise/censoring:** perturb line fluxes before forming ratios; propagate line-error distributions through predictions [@Shy2022MeasurementError].
4. **Ablations:** compare BPT-only lines, all strong optical lines, no single high-leverage line, and no density/shock-sensitive lines [@Baldwin1981BPT; @Veilleux1987VO; @Allen2008ShockGrids].
5. **Uncertainty:** calibrate probabilities on validation data separate from model training; report calibration curves/ECE/Brier score rather than only ROC AUC [scikit-learn User Guide: https://scikit-learn.org/stable/modules/calibration.html; @Guo2017Calibration].
6. **OOD detection:** treat shocks, LIER/retired-galaxy models, and observed composites as “do not know” tests [@Allen2008ShockGrids; @Stasinska2008Retired; @CidFernandes2011WHAN; @Belfiore2016LIER; @DeSouza2017ProbELG].
7. **Observational check:** apply the trained model to survey samples and analyze disagreements with BPT, WHAN, WISE mid-IR, and radio indicators; add X-ray/UV only after adding dedicated AGN-selection references [@Kauffmann2003AGNHosts; @CidFernandes2011WHAN; @Stern2012WISEAGN; @BestHeckman2012RadioAGN; @DingRodriguez2024Multiwavelength].

The more defensible project angle is not “ML beats BPT everywhere.” A more credible angle is: **ML trained on physical grids can map where the HII/AGN separation is robust, where it is label-dependent, and where uncertainty/abstention is scientifically necessary** [@Kewley2006AGNClassification; @DeSouza2017ProbELG; @Cho2026DangerZone; @Zhu2023NLRPhotoionization].

## 10. Evidence-backed caveats and disagreements

- **BPT remains strong and interpretable.** ML should not replace it by default; it should clarify ambiguous zones, missing-line regimes, or multi-dimensional feature spaces [@Baldwin1981BPT; @Veilleux1987VO; @Kewley2001MaxStarburst; @Kauffmann2003AGNHosts].
- **BPT-derived labels limit novelty.** If labels come from BPT curves, the model is learning a historical diagnostic, not independently discovering ionization physics [@Kewley2006AGNClassification; @Zhang2019MLCELG; @Gupta2024EmissionDNN].
- **Simulation labels are clean but idealized.** Photoionization grids allow controlled labels, but uncertainty in grid physics and parameter coverage must be propagated [@Ferland2017Cloudy; @Levesque2010Starforming; @Zhu2023NLRPhotoionization].
- **Deep learning is not automatically superior.** Dense neural networks and autoencoders can be useful, but the reviewed evidence only supports a narrower claim: random forests performed well in some line-ratio/multi-wavelength studies, while DNN/SVM and representation-learning approaches are also active. Treat tree ensembles as strong baselines, not as proven universally superior models [@Zhang2019MLCELG; @DingRodriguez2024Multiwavelength; @Gupta2024EmissionDNN; @Pat2023ReconstructingSDSS].
- **Cross-wavelength validation is useful but not definitive.** Mid-IR and radio AGN selections select different subsets and can disagree with optical diagnostics for physical reasons. X-ray/UV validation remains a reasonable extension but needs dedicated citations beyond the sources audited here [@Stern2012WISEAGN; @BestHeckman2012RadioAGN; @DingRodriguez2024Multiwavelength].

## Open questions

1. Which line-ratio combinations remain robust across Cloudy and MAPPINGS AGN/HII grids? [@Ferland2017Cloudy; @Allen2008ShockGrids; @Zhu2023NLRPhotoionization]
2. How should composite spectra be labeled: binary, fractional AGN contribution, or abstention? [@Kewley2006AGNClassification; @DeSouza2017ProbELG; @Cho2026DangerZone]
3. Can a classifier trained on synthetic grids produce calibrated probabilities on observed spectra without empirical recalibration? [@Guo2017Calibration; @Humphrey2022PopulationShift]
4. What is the best contamination test set for shocks, LIER/retired galaxies, planetary nebulae, and diffuse ionized gas? [@Allen2008ShockGrids; @Stasinska2008Retired; @Belfiore2016LIER; @Rhea2023EmissionRegionNN]
5. How much does spatial resolution change ML labels relative to integrated/fiber spectra? [@Kewley2005Aperture; @Richards2015SAMIaperture; @Belfiore2016LIER]
6. What external validation signal is most appropriate for weak AGN: mid-IR, radio, IFU nuclear concentration, or additional X-ray/UV/variability indicators that still need dedicated review? [@CidFernandes2011WHAN; @Stern2012WISEAGN; @BestHeckman2012RadioAGN; @DingRodriguez2024Multiwavelength]

## Recommended next steps

1. Build a literature table with columns: paper, data, labels, features, model, validation, metrics, and failure modes [@Zhang2019MLCELG; @DingRodriguez2024Multiwavelength; @Wang2023LAMOSTELG; @Gupta2024EmissionDNN].
2. For any new classifier, pre-register baseline comparisons against BPT/WHAN and grouped physical holdouts [@Kauffmann2003AGNHosts; @Kewley2001MaxStarburst; @CidFernandes2011WHAN; @Zhu2023NLRPhotoionization].
3. Treat PDF/BibTeX collection as the reproducible source bundle for the literature review [see `outputs/ml-galaxy-emission-classification.bib` and `outputs/.sources/ml-galaxy-emission-classification/pdf-manifest.md`].
4. Prioritize uncertainty and domain-shift experiments before model architecture changes [@Shy2022MeasurementError; @Guo2017Calibration; @Humphrey2022PopulationShift].
5. If preparing a paper, position the work as a physics-aware evaluation framework rather than simply another classifier [@Zhu2023NLRPhotoionization; @DeSouza2017ProbELG; @Cho2026DangerZone].

## Sources

Primary literature and documentation cited in the report:

- Baldwin, Phillips & Terlevich (1981), “Classification parameters for the emission-line spectra of extragalactic objects” — https://doi.org/10.1086/130766
- Veilleux & Osterbrock (1987), “Spectral classification of emission-line galaxies” — https://doi.org/10.1086/191166
- Kewley et al. (2001), “Theoretical Modeling of Starburst Galaxies” — https://arxiv.org/abs/astro-ph/0106324
- Kauffmann et al. (2003), “The Host Galaxies of Active Galactic Nuclei” — https://arxiv.org/abs/astro-ph/0304239
- Kewley et al. (2006), “The host galaxies and classification of active galactic nuclei” — https://doi.org/10.1111/j.1365-2966.2006.10859.x
- Cid Fernandes et al. (2011), WHAN classification — https://arxiv.org/abs/1012.4426
- Stasińska et al. (2008), retired galaxies — https://arxiv.org/abs/0809.1341
- Kewley, Nicholls & Sutherland (2019), emission-line review — https://doi.org/10.1146/annurev-astro-081817-051832
- D’Agostino et al. (2019), IFU shock/AGN/SF diagnostic — https://arxiv.org/abs/1902.10295
- Zhang et al. (2019), ML classifiers for ELGs — https://doi.org/10.3847/1538-4357/ab397e and https://github.com/zkdtc/MLC_ELGs
- Teimoorinia & Keown (2018), missing Hα/[N II] ML classification — https://arxiv.org/abs/1805.04069
- Ding & Rodriguez (2024), multi-wavelength supervised ML — https://doi.org/10.1088/1538-3873/ad9b4e
- Gupta et al. (2024), dense NN/SVM ELG classification — https://doi.org/10.3847/2515-5172/ad3422
- Wang et al. (2023), LAMOST ELG spectral ML — https://doi.org/10.1016/j.newast.2022.101965
- de Souza et al. (2017), probabilistic ELG classification — https://arxiv.org/abs/1703.07607
- Cho et al. (2026), BPT buffers with UMAP/HDBSCAN — https://doi.org/10.3847/1538-4357/ae367d
- Pat et al. (2023), full-spectrum representation learning — https://arxiv.org/abs/2211.11783
- Rhea et al. (2023), Cloudy/shock-grid neural-network emission-region classification — https://arxiv.org/abs/2306.11545
- Belfiore et al. (2016), MaNGA LIERs — https://arxiv.org/abs/1605.07189
- Belfiore et al. (2019), MaNGA DAP emission-line modeling — https://arxiv.org/abs/1901.00866
- Kewley et al. (2005), aperture effects — https://arxiv.org/abs/astro-ph/0501229
- Richards et al. (2015), SAMI aperture corrections — https://arxiv.org/abs/1510.06038
- Ferland et al. (2017), Cloudy C17 — https://arxiv.org/abs/1705.10877
- Levesque et al. (2010), star-forming grids — https://arxiv.org/abs/0908.0460
- Allen et al. (2008), MAPPINGS shock grids — https://arxiv.org/abs/0805.0204
- Zhu et al. (2023), AGN NLR photoionization models — https://arxiv.org/abs/2305.12670
- Meléndez et al. (2014), AGN diagnostics — https://arxiv.org/abs/1406.5563
- Shy et al. (2022), measurement error in astronomical classification — https://doi.org/10.3847/1538-3881/ac6e64
- Humphrey et al. (2022), population-shift performance estimation — https://arxiv.org/abs/2209.15112
- Guo et al. (2017), probability calibration — https://arxiv.org/abs/1706.04599
- SDSS MPA-JHU galaxy properties documentation — https://www.sdss3.org/dr9/algorithms/galaxy_mpa_jhu.php
- MaNGA DAP emission-line documentation — https://sdss-mangadap.readthedocs.io/en/latest/emissionlines.html
- CALIFA DR3 — https://califa.caha.es/CALIFA_3rd_DATA_RELEASE.html
- SAMI DR3 — https://sami-survey.org/node/902
- scikit-learn probability calibration documentation — https://scikit-learn.org/stable/modules/calibration.html

- Fraix-Burnet et al. (2021), unsupervised SDSS spectral classes — https://arxiv.org/abs/2103.05928
- Teimoorinia et al. (2022), DESOM unsupervised MaNGA spectral maps — https://arxiv.org/abs/2112.03425
- Melchior et al. (2022), SPENDER autoencoding galaxy spectra I — https://arxiv.org/abs/2211.07890
- Liang et al. (2023), SPENDER outlier detection with normalizing flows — https://arxiv.org/abs/2302.02496
- Alcolea et al. (2026), semi-supervised DESI AGN discovery with autoencoder representations — https://arxiv.org/abs/2607.07329
- Parker et al. (2024), AstroCLIP cross-modal self-supervised galaxy image/spectrum embeddings — https://arxiv.org/abs/2310.03024
- Pattnaik et al. (2025), SpecPT transformer for spectra reconstruction and redshift measurement — https://arxiv.org/abs/2501.01070
- Khederlarian et al. (2024), differentiable emission-line equivalent-width prediction — https://arxiv.org/abs/2404.03055

The companion BibTeX file is `outputs/ml-galaxy-emission-classification.bib`; the PDF manifest is `outputs/.sources/ml-galaxy-emission-classification/pdf-manifest.md`.

## Revision note after verification pass

A reviewer-style verification pass flagged four major issues: overbroad tree/ensemble claims, overly categorical synthetic-grid domain-shift language, under-cited X-ray/UV validation language, and vague source-type notes for papers without local PDFs. The revised text narrows those claims, marks domain shift as a design risk rather than a directly measured result, removes strong X-ray/UV validation claims, and adds explicit source-type caveats.

## Source and URL verification notes

- BibTeX keys used above come from `outputs/ml-galaxy-emission-classification.bib`; downloaded PDF availability was checked against `outputs/.sources/ml-galaxy-emission-classification/pdf-manifest.md`.
- The manifest reports downloaded PDFs for most core sources, including Baldwin, Veilleux, Kewley 2001/2005/2006, Kauffmann 2003, Cid Fernandes 2011, Stasińska 2006/2008, Belfiore 2016/2019, D’Agostino 2019, de Souza 2017, Ding & Rodriguez 2024, Rhea 2023, Ferland 2017, Allen 2008, Levesque 2010, Zhu 2023, Guo 2017, Humphrey 2022, and others.
- Source-type caveat for papers without local PDFs: Zhang 2019 was supported by DOI metadata/HTML notes plus the public GitHub repository; Gupta 2024 was supported by DOI/HTML notes; Cho 2026 was supported by DOI/HTML notes; Shy 2022 was supported by DOI metadata/abstract-level evidence; Kewley, Nicholls & Sutherland 2019 was supported by DOI/review metadata and research notes; Wang 2023 was supported by DOI/ScienceDirect abstract and section-snippet evidence, not a locally downloaded full text. Claims depending on these sources are kept at a summary level.
- URL checks were performed with `curl -L -I` where possible. ArXiv URLs and official documentation URLs for SDSS MPA-JHU, MaNGA DAP, CALIFA DR3, and SAMI DR3 returned HTTP 200. Several publisher DOI targets resolved but returned anti-bot/security or access-control pages (notably OUP 403 responses and IOP/ADS validation pages), so those should be treated as resolver-confirmed but not fully content-validated by curl.
