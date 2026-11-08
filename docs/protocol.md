---
title: "Pairwise perception survey: pre-registration-style protocol (draft v0.1)"
---

**Status: not run.** No responses have been collected from anyone. Everything
below that looks like a result comes from simulated respondents. The survey
page in this repository is a demonstration that stores nothing and sends
nothing. Independent personal project, not affiliated with the
Singapore-ETH Centre or the NUS Urban Analytics Lab.

## 1. Question

At a given built density, do street scenes with more visible greenery (green
view index, GVI) or more visible sky (sky view index, SVI) *feel* less
crowded? If yes, greenery and sky openness are candidate design levers for
"softening" density, which is what the street-level image analysis in this
repository (README, Tier A and Tier B) can only show as association.

## 2. Hypotheses (directional, stated before any data)

* **H1 (primary).** Controlling for the scene's density proxy, higher GVI
  raises the probability of a scene being chosen as "feels less crowded"
  (β_GVI > 0 in the second-stage regression, one-sided α = 0.05 reported as
  two-sided p < 0.05 with the predicted sign).
* **H2 (secondary).** Controlling for density, higher SVI raises that
  probability (β_SVI > 0).
* **H3 (secondary).** Higher density proxy (log(1 + GFA intensity) of the
  scene's 250 m cell) lowers it (β_density < 0).
* Exploratory, not confirmatory: interactions GVI × density, and agreement of
  the human ranking with the Place Pulse-style perception predictions shipped
  in Global Streetscapes.

## 3. Stimuli

* <!--n:n_stimuli-->40<!--/n--> scenes from the island-wide KartaView sample
  (CC BY-SA 4.0), only photos that KartaView's own pipeline reports as
  face/plate-blurred, stratified into 4 density quartiles × 3 GVI tertiles
  so that GVI and density are not collinear in the stimulus set
  (`scripts/10_survey_design.py`, `docs/survey/stimuli.json`).
* GVI/SVI of each stimulus come from our Mask2Former re-segmentation, which
  reproduces the Global Streetscapes values on 477 overlapping images (README).
* Known limitations of the stimulus set: dashcam viewpoint (road, not
  pavement), mixed capture years (2015–2026), weather and time of day vary.

## 4. Task and design

* Two-alternative forced choice: "Which street feels less crowded?"
* Each respondent sees 20 pairs: 19 random distinct pairs plus one repeated
  pair with sides swapped (within-respondent consistency check). Left/right
  position randomised.
* No identifying data: no names, e-mails or IP logging; optional coarse
  demographics (age band, years lived in Singapore, housing type) only in a
  real deployment, after ethics review.

## 5. Sample size (simulation-based power)

Generative model for the simulation: latent utility of scene *i* =
β·z(GVI_i) − 0.3·z(density_i) + u_i, with u_i ~ N(0, 0.5²) for scene-level
appeal not explained by GVI or density, rater discrimination
1 + N(0, 0.3²), Bradley–Terry choice probabilities, 200 simulated studies per
cell. The analysis replicated in each simulated study is exactly the planned
one (§6). The scene-level term u_i means that power is capped by the number
of scenes, not only by the number of respondents.

| β (logit per SD of GVI) | 25 respondents | 50 | 100 | 200 |
|---|---|---|---|---|
| 0 (false-positive rate) | <!--n:power_e00_n25-->0%<!--/n--> | <!--n:power_e00_n50-->0%<!--/n--> | <!--n:power_e00_n100-->0%<!--/n--> | <!--n:power_e00_n200-->0%<!--/n--> |
| 0.20 | <!--n:power_e20_n25-->0%<!--/n--> | <!--n:power_e20_n50-->0%<!--/n--> | <!--n:power_e20_n100-->0%<!--/n--> | <!--n:power_e20_n200-->0%<!--/n--> |
| 0.35 | <!--n:power_e35_n25-->0%<!--/n--> | <!--n:power_e35_n50-->0%<!--/n--> | <!--n:power_e35_n100-->0%<!--/n--> | <!--n:power_e35_n200-->0%<!--/n--> |
| 0.50 | <!--n:power_e50_n25-->0%<!--/n--> | <!--n:power_e50_n50-->0%<!--/n--> | <!--n:power_e50_n100-->0%<!--/n--> | <!--n:power_e50_n200-->0%<!--/n--> |

**Stopping rule (planned):** recruit 100 complete respondents, analyse once.
Neither the effect size nor the scene-level SD is known, so the first step of
a real study would be a small pilot (about 20 respondents) used *only* to
re-estimate the scene-level SD and update this table, not to test H1–H3.

## 6. Analysis plan

1. Exclusions (decided now): respondents with median response time < 400 ms
   or who choose the same side in ≥ 18 of 20 pairs. The repeated pair is
   reported, not used for exclusion.
2. Stage 1: Bradley–Terry scores per scene, pooled over respondents
   (Hunter's MM algorithm with 0.5 virtual wins/losses against a reference,
   `ssd/pairwise.py`). TrueSkill scores computed as a robustness check.
3. Stage 2: OLS of the 40 scene scores on z(GVI), z(SVI) and z(density).
   Inference at the scene level (n = 40), which is conservative with respect to
   the respondent count. H1 is supported if β_GVI > 0 with p < 0.05.
4. Robustness: (a) Bradley–Terry logit with scene covariates and a
   respondent random effect; (b) leave-one-planning-area-out on scenes;
   (c) excluding scenes with rain or night lighting.

## 7. Ethics

* A real deployment requires review by an institutional review board before
  any recruitment, an information sheet, and an explicit consent step. Adults
  only.
* Responses would be anonymous; only choice data and coarse optional
  demographics would be kept; no free text.
* Images: KartaView photos under CC BY-SA 4.0, credited per image; only
  photos the platform reports as blurred are shown. Scenes may still contain
  identifiable buildings and vehicles; respondents are not asked about people.
* The demo page in this repository does not collect or transmit anything.
