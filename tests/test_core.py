import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from ssd import agreement, density, grid, indices, pairwise, sampling, spatial

VISTAS = {0: "Bird", 6: "Wall", 17: "Building", 27: "Sky", 29: "Terrain", 30: "Vegetation",
          63: "Car Mount", 64: "Ego Vehicle"}
CITYSCAPES = {0: "road", 2: "building", 3: "wall", 8: "vegetation", 9: "terrain", 10: "sky"}


# ------------------------------------------------------------------ grid
def test_cell_id_floor_and_negative():
    assert grid.cell_id(np.array([0.0, 249.9, 250.0]), np.array([0.0, 0.0, 499.0])).tolist() == ["0_0", "0_0", "1_1"]
    assert grid.cell_id(np.array([-1.0]), np.array([10.0])).tolist() == ["-1_0"]


def test_cell_bounds_roundtrip():
    assert grid.cell_bounds("3_4") == (750.0, 1000.0, 1000.0, 1250.0)
    assert grid.block_id(np.array([1999.0, 2000.0]), np.array([0.0, 0.0])).tolist() == ["0_0", "1_0"]


# ------------------------------------------------------------------ indices
def test_category_ids_vistas_and_cityscapes():
    v = indices.category_ids(VISTAS)
    assert v == {"vegetation": [30], "sky": [27], "building": [6, 17]}
    c = indices.category_ids(CITYSCAPES)
    assert c == {"vegetation": [8], "sky": [10], "building": [2, 3]}
    assert indices.excluded_ids(VISTAS) == [63, 64]


def test_terrain_is_not_vegetation():
    lm = np.array([[30, 29], [27, 17]])
    s = indices.shares(lm, indices.category_ids(VISTAS), n_classes=65)
    assert s == {"vegetation": 0.25, "sky": 0.25, "building": 0.25, "labelled": 1.0}


def test_shares_denominator_and_exclusion():
    lm = np.array([30, 30, 27, 64, 64, 0, 0, 0])
    cats = indices.category_ids(VISTAS)
    s = indices.shares(lm, cats, n_classes=65)
    assert s["vegetation"] == pytest.approx(2 / 8)
    s2 = indices.shares(lm, cats, exclude=[63, 64], n_classes=65)
    assert s2["vegetation"] == pytest.approx(2 / 6)
    assert s2["labelled"] == pytest.approx(6 / 8)


def test_norm_and_missing_category():
    assert indices.norm("construction--structure--Building") == "building"
    assert indices.norm("Traffic Sign (Back)") == "traffic-sign-(back)"
    with pytest.raises(ValueError):
        indices.category_ids({0: "road", 1: "sky"})


# ------------------------------------------------------------------ agreement
def test_agreement_perfect_and_biased():
    a = np.array([0.1, 0.2, 0.3, 0.5, 0.9])
    r = agreement.agreement(a, a)
    assert r["ccc"] == pytest.approx(1.0) and r["mae"] == 0 and r["bias"] == 0
    r2 = agreement.agreement(a + 0.1, a)
    assert r2["pearson_r"] == pytest.approx(1.0)
    assert r2["ccc"] < 0.95 and r2["bias"] == pytest.approx(0.1)
    assert r2["loa_low"] == pytest.approx(0.1) and r2["mae"] == pytest.approx(0.1)


def test_ccc_known_value():
    a = np.array([1.0, 2, 3, 4])
    b = np.array([2.0, 3, 4, 5])
    # cov = 1.25, var = 1.25 each, mean diff = 1 -> 2.5 / (2.5 + 1)
    assert agreement.lins_ccc(a, b) == pytest.approx(2.5 / 3.5)


def test_agreement_drops_nan_and_needs_three():
    r = agreement.agreement([0.1, np.nan, 0.3, 0.4, 0.2], [0.1, 0.2, 0.3, 0.4, 0.25])
    assert r["n"] == 4
    with pytest.raises(ValueError):
        agreement.agreement([1, 2], [1, 2])


# ------------------------------------------------------------------ spatial
def test_morans_i_signs():
    xy = np.array([(x, y) for x in range(10) for y in range(10)], float)
    w = spatial.knn_weights(xy, k=4)
    assert np.allclose(w.sum(1), 1.0) and np.all(np.diag(w) == 0)
    smooth = xy[:, 0]
    checker = ((xy[:, 0] + xy[:, 1]) % 2) * 2 - 1
    assert spatial.morans_i(smooth, w) > 0.6
    assert spatial.morans_i(checker, w) < -0.6
    i, p = spatial.morans_i_perm(smooth, w, n_perm=99)
    assert p <= 0.02


def test_blocked_folds_keep_blocks_together():
    blocks = np.array(["a", "a", "b", "c", "c", "d", "e", "e", "f"])
    f = spatial.blocked_folds(blocks, n_folds=3, seed=1)
    for b in np.unique(blocks):
        assert len(set(f[blocks == b])) == 1
    assert set(f) == {0, 1, 2}


def test_cv_r2_mean_model_near_zero_and_ols_high():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 1))
    y = 2 * X[:, 0] + rng.normal(0, 0.1, 200)
    folds = rng.integers(0, 5, 200)
    from ssd.models import mean_fit_predict, ols_fit_predict
    assert spatial.cv_r2(X, y, folds, ols_fit_predict) > 0.99
    assert spatial.cv_r2(X, y, folds, mean_fit_predict) < 0.02


# ------------------------------------------------------------------ density
def test_parse_gpr():
    assert density.parse_gpr("2.8") == 2.8
    for v in ["EVA", "SDP", "LND", None, "0", "-1"]:
        assert math.isnan(density.parse_gpr(v))
    assert density.parse_gpr("LND", landed=1.4) == 1.4
    assert math.isnan(density.parse_gpr("EVA", landed=1.4))


def test_landuse_metrics_weighted():
    cells = density.cells_frame(["0_0"])
    lu = gpd.GeoDataFrame({"LU_DESC": ["RESIDENTIAL", "COMMERCIAL", "PARK"], "GPR": ["4.0", "1.0", None]},
                          geometry=[box(0, 0, 125, 250), box(125, 0, 250, 125), box(125, 125, 250, 250)],
                          crs=3414)
    m = density.landuse_metrics(cells, lu, landed=None).loc["0_0"]
    assert m.gpr_mean == pytest.approx((4 * 0.5 + 1 * 0.25) / 0.75)
    assert m.gfa_intensity == pytest.approx(4 * 0.5 + 1 * 0.25)
    assert m.gpr_area_share == pytest.approx(0.75)
    assert m.res_share == pytest.approx(0.5) and m.park_share == pytest.approx(0.25)


def test_coverage_clips_to_cell():
    cells = density.cells_frame(["0_0", "1_0"])
    fp = gpd.GeoDataFrame(geometry=[box(200, 0, 300, 50)], crs=3414)
    c = density.coverage(cells, fp, "cov")
    assert c["0_0"] == pytest.approx(50 * 50 / 62500)
    assert c["1_0"] == pytest.approx(50 * 50 / 62500)


# ------------------------------------------------------------------ pairwise
def test_simulate_shape_and_direction():
    rng = np.random.default_rng(0)
    c = pairwise.simulate(np.array([3.0, -3.0]), 50, 10, rng)
    assert c.shape == (500, 4)
    wins_item0 = np.mean(np.where(c[:, 3] == 1, c[:, 1], c[:, 2]) == 0)
    assert wins_item0 > 0.9


def test_bradley_terry_recovers_scale():
    rng = np.random.default_rng(3)
    lat = rng.normal(0, 1, 30)
    lat -= lat.mean()
    c = pairwise.simulate(lat, 200, 30, rng)
    s = pairwise.bradley_terry(c, 30)
    assert abs(s.mean()) < 1e-9
    slope = np.polyfit(lat, s, 1)[0]
    assert 0.85 < slope < 1.15
    assert np.corrcoef(lat, s)[0, 1] > 0.97


def test_bradley_terry_finite_for_undefeated():
    c = np.array([[0, 0, 1, 1]] * 5 + [[0, 1, 2, 1]] * 5)
    s = pairwise.bradley_terry(c, 3)
    assert np.all(np.isfinite(s)) and s[0] > s[1] > s[2]


def test_trueskill_orders_and_shrinks_sigma():
    rng = np.random.default_rng(5)
    lat = np.linspace(-2, 2, 10)
    c = pairwise.simulate(lat, 100, 20, rng)
    mu, sd = pairwise.trueskill(c, 10)
    assert np.corrcoef(lat, mu)[0, 1] > 0.95
    assert np.all(sd < 25 / 3)
    mu1, _ = pairwise.trueskill(np.array([[0, 0, 1, 1]]), 2)
    assert mu1[0] > 25 > mu1[1]


def test_slope_test_detects_and_nulls():
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    b, p = pairwise.slope_test(0.5 * x + rng.normal(0, 0.2, 100), x)
    assert b > 0.4 and p < 1e-10
    b0, p0 = pairwise.slope_test(rng.normal(size=100), x)
    assert p0 > 0.01


def test_power_monotone_and_null_size():
    x = np.random.default_rng(1).normal(size=40)
    p_null = pairwise.power(x, 0.0, 0.5, 30, 15, n_sims=60, seed=1)
    p_big = pairwise.power(x, 0.6, 0.3, 30, 15, n_sims=30, seed=1)
    assert p_null < 0.12
    assert p_big > 0.9


def test_landed_parcels_count_toward_intensity():
    cells = density.cells_frame(["0_0"])
    lu = gpd.GeoDataFrame({"LU_DESC": ["RESIDENTIAL"], "GPR": ["LND"]}, geometry=[box(0, 0, 250, 125)], crs=3414)
    assert density.landuse_metrics(cells, lu, landed=1.4).loc["0_0"].gfa_intensity == pytest.approx(0.7)
    assert density.landuse_metrics(cells, lu, landed=None).loc["0_0"].gfa_intensity == 0.0


# ------------------------------------------------------------------ number check
def test_number_check_catches_wrong_and_unknown(tmp_path):
    import shutil
    from pathlib import Path

    from ssd import claims
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "outputs", tmp_path / "outputs")
    reg = claims.registry(tmp_path)
    key = "agree_gvi_gs_vs_m2f_ccc"
    good = f"CCC <!--n:{key}-->{reg[key]}<!--/n-->"
    (tmp_path / "paper").mkdir()
    (tmp_path / "README.md").write_text(good, encoding="utf-8")
    (tmp_path / "paper" / "paper.md").write_text(good, encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "protocol.md").write_text(good, encoding="utf-8")
    assert claims.check_numbers(tmp_path) == []
    (tmp_path / "README.md").write_text(good + " <!--n:agree_n-->9999<!--/n--> <!--n:nope-->1<!--/n-->",
                                        encoding="utf-8")
    errs = claims.check_numbers(tmp_path)
    assert len(errs) == 2 and any("agree_n" in e for e in errs) and any("unknown key nope" in e for e in errs)


def test_registry_values_follow_outputs(tmp_path):
    import json
    import shutil
    from pathlib import Path

    from ssd import claims
    root = Path(__file__).resolve().parents[1]
    shutil.copytree(root / "outputs", tmp_path / "outputs")
    ag = json.loads((tmp_path / "outputs" / "agreement.json").read_text())
    reg = claims.registry(tmp_path)
    assert reg["agree_gvi_gs_vs_m2f_ccc"] == f"{ag['gvi']['gs_vs_m2f']['ccc']:.3f}"
    assert reg["agree_svi_gs_vs_segf_mae"] == f"{ag['svi']['gs_vs_segf']['mae']:.3f}"


# ------------------------------------------------------------------ sampling
def test_random_prefix_stops_at_first_gap_and_is_seeded():
    ids = [f"id{k:03d}" for k in range(50)]
    order = sampling.random_prefix(ids, set(ids), seed=3)
    assert len(order) == 50 and order != sorted(ids)
    assert sampling.random_prefix(ids, set(ids), seed=3) == order
    done = set(order[:10]) | set(order[11:30])      # gap at position 10
    assert sampling.random_prefix(ids, done, seed=3) == order[:10]
    assert sampling.random_prefix(list(reversed(ids)), done, seed=3) == order[:10]


# ------------------------------------------------------------------ added after mutation run
def test_bland_altman_limits_and_mae_use_absolute_differences():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([2.0, 1.0, 4.0, 3.0])          # differences -1, +1, -1, +1
    r = agreement.agreement(a, b)
    sd = np.std(a - b, ddof=1)
    assert r["bias"] == pytest.approx(0.0)
    assert r["mae"] == pytest.approx(1.0)
    assert r["loa_high"] == pytest.approx(1.96 * sd)
    assert r["loa_low"] == pytest.approx(-1.96 * sd)


def test_cv_r2_does_not_train_on_test_fold():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 1))
    y = rng.normal(size=120)                    # no signal at all
    folds = np.arange(120) % 4

    def nearest_neighbour(Xtr, ytr, Xte):      # memorises its training rows
        d = np.abs(Xte[:, None, 0] - Xtr[None, :, 0])
        return ytr[d.argmin(1)]
    assert spatial.cv_r2(X, y, folds, nearest_neighbour) < 0.2
