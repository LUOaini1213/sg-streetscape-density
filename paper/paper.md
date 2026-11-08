---
title: "Street-level greenery, sky and built density in Singapore: a reproducible baseline and a perception-survey design"
author: "Wenjie Luo (independent personal project; not affiliated with the Singapore-ETH Centre or NUS Urban Analytics Lab)"
date: "November 2026 (working paper, not peer reviewed)"
---

## Abstract

We ask how street-level visual features vary with built density in Singapore,
and how one would test whether greenery makes density feel less crowded.

* **Reproduction.** We re-segmented <!--n:agree_n-->477<!--/n--> KartaView
  images from the open Global Streetscapes dataset with Mask2Former (Mapillary
  Vistas). The provided green view index (GVI) was reproduced with Lin's CCC
  <!--n:agree_gvi_gs_vs_m2f_ccc-->0.993<!--/n--> and the sky view index (SVI)
  with <!--n:agree_svi_gs_vs_m2f_ccc-->0.999<!--/n-->.
* **Model sensitivity.** A SegFormer/Cityscapes model agrees on GVI and SVI
  but less so on building share (CCC
  <!--n:agree_building_share_gs_vs_segf_ccc-->0.863<!--/n-->).
* **Coverage.** Global Streetscapes covers only a ~2.4 km window of
  Singapore. We therefore also segmented an island-wide sample of
  <!--n:isl_images-->2,580<!--/n--> public KartaView photos.
* **Density link.** Planned floor-area intensity from the URA Master Plan 2019
  is associated with more building façade in view. In the dataset window it
  is also associated with less greenery (mixed-model slope
  <!--n:win_gvi_adjusted_b-->-0.080<!--/n--> GVI per unit log intensity).
  Park land confounds this strongly.
* **Survey design.** We specify a pairwise-comparison survey with Bradley–Terry
  scoring and a simulated power analysis. No responses have been collected.

## 1. Introduction

High-density housing is central to Singapore's land-use strategy. Whether it
*feels* dense depends partly on what people see at street level: trees, sky,
façades. Street-view imagery with semantic segmentation has become the
standard way to measure that view at scale (e.g. GVI and SVI). Large open
datasets such as Global Streetscapes (Hou et al., 2024) now ship these
measures pre-computed.

Before such measures are linked to density or used to pick survey stimuli,
three questions matter. Can we reproduce them? How sensitive are they to the
model? How do they relate to density once obvious confounders are handled?
This note answers the first two directly. It gives a cautious, confounder-aware
answer to the third. For the perceptual question that images alone cannot
answer, it designs (but does not run) a survey.

## 2. Data

**Global Streetscapes.** We retrieved all Singapore rows from the Hugging Face
release (CC BY-SA 4.0) with DuckDB HTTP range reads: segmentation pixel
counts, GVI/SVI, contextual labels, perception predictions and OSM road
attributes. There are <!--n:gs_sg_images-->22,327<!--/n--> images. All of them
lie within ~2.4 km of the city reference point, covering parts of Tanglin,
Queenstown and Bukit Timah. Mapillary images need a token and were not
downloaded. KartaView images are public.

**Island-wide KartaView sample.** We queried the public KartaView API on a
1 km hexagonal lattice with a 500 m search radius. That returned
<!--n:isl_api_photos-->12,804<!--/n--> unique photos. We kept those taken
08:00–17:59 local time, in 2015 or later, and non-panoramic, then took at most
two per 250 m cell from distinct sequences. Result:
<!--n:isl_images-->2,580<!--/n--> photos in <!--n:isl_cells-->1,874<!--/n-->
cells and <!--n:isl_pas-->53<!--/n--> planning areas.

**Density proxies** come from data.gov.sg under the Singapore Open Data
Licence:

* URA Master Plan 2019 Land Use, with parcel GPR and land-use class;
* HDB Existing Building footprints.

We did not use OSM footprints, to avoid mixing ODbL and CC BY-SA in
redistributed derived tables.

## 3. Methods

**Segmentation.** We ran inference only, on 1280×720 thumbnails in fp16
(GTX 1650, 4 GB). Two models were used:
`mask2former-swin-large-mapillary-vistas-semantic` (65 classes, identical to
the dataset's label set) and `segformer-b2-finetuned-cityscapes` (19 classes).
GVI = vegetation / all pixels and SVI = sky / all pixels, following the
dataset's definition (its denominator includes ego-vehicle pixels). Building
share = (building + wall) / all pixels. Agreement is summarised with Pearson's
r, Lin's concordance correlation coefficient (CCC), mean absolute error and
Bland–Altman limits.

**Spatial unit and density.** We used a 250 m square grid in SVY21. The GFA
intensity of a cell is Σ(GPR × parcel area) / cell area. Landed-housing
parcels ("LND", which have no numeric GPR) are assigned 1.4; a sensitivity run
excludes them. Parcels with GPR "EVA" or "SDP", parks and roads contribute
zero. Covariates: park/open-space, residential and road land shares, and HDB
footprint coverage.

**Models.**

* *Tier A* (dataset window, provided attributes): image-level linear mixed
  models with a random intercept per cell. Fixed effects are log(1 + GFA
  intensity), land-use shares and image covariates: walking vs driving
  platform, source, side view, major road and year. Robustness checks are
  image-level OLS with cell-clustered SEs and the no-landed variant.
* *Tier B* (island-wide, our values): image-level OLS with SEs clustered by
  2 km block.
* *Both tiers:* cell-level OLS feature sets were compared by spatially blocked
  5-fold CV (whole 500 m or 2 km blocks held out, 10 fold assignments). We
  report residual Moran's I on 6-nearest-neighbour weights with permutation
  p-values.

**Survey instrument.** A static web page shows two scenes and asks which
"feels less crowded". The stimuli are <!--n:n_stimuli-->40<!--/n-->
face/plate-blurred scenes stratified by density quartile × GVI tertile. Each
respondent gets 20 trials, including one swapped repeat. The planned analysis
has two stages:

1. Bradley–Terry scores (Hunter's MM algorithm with a weak prior), with
   TrueSkill as a check.
2. OLS of the scene scores on z(GVI) and z(density), with inference at the
   scene level.

Power was estimated by simulating that exact pipeline under the
pre-registered generative model.

## 4. Results

**Reproduction and model sensitivity.** Against the dataset values, our
Mask2Former run gives CCC <!--n:agree_gvi_gs_vs_m2f_ccc-->0.993<!--/n--> for
GVI (MAE <!--n:agree_gvi_gs_vs_m2f_mae-->0.012<!--/n-->),
<!--n:agree_svi_gs_vs_m2f_ccc-->0.999<!--/n--> for SVI and
<!--n:agree_building_share_gs_vs_m2f_ccc-->0.973<!--/n--> for building share.
SegFormer gives <!--n:agree_gvi_gs_vs_segf_ccc-->0.981<!--/n-->,
<!--n:agree_svi_gs_vs_segf_ccc-->0.991<!--/n--> and
<!--n:agree_building_share_gs_vs_segf_ccc-->0.863<!--/n--> respectively, with
a building-share bias of
<!--n:agree_building_share_gs_vs_segf_bias-->+0.027<!--/n-->. GVI and SVI are
therefore robust to model choice. Building share is model-specific and should
not be pooled across models.

![Agreement between provided and re-computed indices (477 images).](figures/agreement.png)

**Density, Tier A** (<!--n:win_images_used-->22,104<!--/n--> images,
<!--n:win_cells-->110<!--/n--> cells).

* *Building share.* The adjusted mixed-model slope on log(1 + GFA intensity)
  is <!--n:win_building_share_adjusted_b-->+0.063<!--/n-->
  <!--n:win_building_share_adjusted_ci-->[+0.033, +0.092]<!--/n-->.
* *GVI.* The slope is <!--n:win_gvi_adjusted_b-->-0.080<!--/n-->
  <!--n:win_gvi_adjusted_ci-->[-0.139, -0.021]<!--/n-->. It weakens to
  <!--n:win_gvi_adjusted_ols_clustered_b-->-0.009<!--/n-->
  <!--n:win_gvi_adjusted_ols_clustered_ci-->[-0.089, +0.071]<!--/n--> when images
  rather than cells carry equal weight. The negative GVI association is thus
  driven by between-cell contrasts and is not robust to the weighting.
* *SVI.* No association: <!--n:win_svi_adjusted_b-->-0.012<!--/n-->
  <!--n:win_svi_adjusted_ci-->[-0.045, +0.021]<!--/n-->.
* *Out of sample.* Density alone explains
  <!--n:win_building_share_cv_density-->0.28<!--/n--> of held-out variance in
  building share but only <!--n:win_gvi_cv_density-->0.13<!--/n--> in GVI.
  Land-use and image controls raise GVI to
  <!--n:win_gvi_cv_density_controls-->0.37<!--/n-->.
* *Park confounding.* Density and park share correlate at
  r = <!--n:win_corr_park-->-0.69<!--/n-->.

**Density, Tier B** (<!--n:isl_images-->2,580<!--/n--> images). The adjusted
slopes are <!--n:isl_building_share_adjusted_b-->+0.000<!--/n-->
<!--n:isl_building_share_adjusted_ci-->[+0.000, +0.000]<!--/n--> for building
share, <!--n:isl_gvi_adjusted_b-->+0.000<!--/n-->
<!--n:isl_gvi_adjusted_ci-->[+0.000, +0.000]<!--/n--> for GVI and
<!--n:isl_svi_adjusted_b-->+0.000<!--/n-->
<!--n:isl_svi_adjusted_ci-->[+0.000, +0.000]<!--/n--> for SVI. In blocked CV,
density alone reaches R² <!--n:isl_building_share_cv_density-->0.00<!--/n-->
for building share and <!--n:isl_gvi_cv_density-->0.00<!--/n--> for GVI.

![Cell mean GVI against planned floor-area intensity, coloured by park share.](figures/density_gvi.png)

**Softened-density candidates.** These are top-quartile-intensity cells that
are greener than density predicts: <!--n:win_soft_n-->15<!--/n--> of
<!--n:win_soft_hi-->28<!--/n--> in Tier A and <!--n:isl_soft_n-->0<!--/n--> of
<!--n:isl_soft_hi-->0<!--/n--> in Tier B (the latter also requiring
median-or-better sky view). They are listed for visual inspection as possible
design precedents. They are not established precedents.

**Survey power (simulated).** With a scene-level SD of 0.5 and 20 pairs per
respondent, an effect of 0.35 logit per SD of GVI is detected with probability
<!--n:power_e35_n50-->0%<!--/n--> at 50 respondents and
<!--n:power_e35_n100-->0%<!--/n--> at 100. At β = 0.2, power is
<!--n:power_e20_n100-->0%<!--/n--> at 100 and
<!--n:power_e20_n200-->0%<!--/n--> at 200 respondents. The false-positive rate
under β = 0 is <!--n:power_e00_n100-->0%<!--/n--> at 100 respondents. Because
scene-level appeal is shared by all respondents, power saturates with the
number of scenes rather than growing without bound with respondents.

![Simulated power of the planned two-stage analysis.](figures/power.png)

## 5. Limitations

* *Coverage and selection.* Global Streetscapes covers one 2.4 km window of
  Singapore. KartaView imagery is volunteered dashcam footage concentrated on
  roads, and the island-wide sample has at most two images per cell.
* *Density measurement.* GPR is a planning control, not built floor area. The
  landed-housing GPR of 1.4 is an assumption. "EVA" parcels (schools, civic
  uses) are treated as zero.
* *Confounding.* Park land, road hierarchy, capture year and camera all
  co-vary with density. The models are associational.
* *Segmentation.* We used thumbnails rather than full-resolution images.
  Building share is model-dependent.
* *Perception.* The dataset's perception scores are model predictions trained
  on Place Pulse, not Singaporean respondents, and our survey has not been run.

## 6. Ethics note

No human-subject data were collected. Any deployment of the survey requires
institutional ethics approval, informed consent and anonymous data handling
(protocol §7). The demo page transmits nothing. Imagery is CC BY-SA 4.0
volunteered content. Survey stimuli are restricted to photos the platform
reports as face/plate-blurred, and derived tables contain no images or
personal data.

## References

* Hou Y, Quintana M, Khomiakov M, Yap W, Ouyang J, Ito K, Wang Z, Zhao T,
  Biljecki F (2024). Global Streetscapes — a comprehensive dataset of 10
  million street-level images across 688 cities for urban science and
  analytics. *ISPRS J. Photogramm. Remote Sens.* 215:216–238.
* Hunter DR (2004). MM algorithms for generalized Bradley–Terry models.
  *Annals of Statistics* 32(1):384–406.
* Herbrich R, Minka T, Graepel T (2007). TrueSkill: a Bayesian skill rating
  system. *NeurIPS 19*.
* Cheng B, Misra I, Schwing AG, Kirillov A, Girdhar R (2022). Masked-attention
  mask transformer for universal image segmentation. *CVPR*.
* Xie E, Wang W, Yu Z, Anandkumar A, Alvarez JM, Luo P (2021). SegFormer.
  *NeurIPS 34*.
* Lin LI (1989). A concordance correlation coefficient to evaluate
  reproducibility. *Biometrics* 45(1):255–268.
