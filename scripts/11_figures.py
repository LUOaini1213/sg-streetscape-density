"""Figures for README / paper and the GeoJSON layer for the Pages map."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pyproj import Transformer  # noqa: E402

FIG = Path("figures")
INK, MUTED, A, B = "#1d1f1c", "#6b7066", "#2f6f4f", "#b5651d"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})


def agreement_fig():
    d = pd.read_csv("outputs/agreement_images.csv")
    ag = json.loads(Path("outputs/agreement.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(9, 3.1))
    for ax, (k, gs, lab) in zip(axs, [("gvi", "gs_gvi", "Green view index"), ("svi", "gs_svi", "Sky view index"),
                                      ("building_share", "gs_building_share", "Building + wall share")]):
        ax.scatter(d[gs], d[f"{k}_m2f"], s=6, color=A, alpha=.6, label="Mask2Former (ours)")
        ax.scatter(d[gs], d[f"{k}_segf"], s=6, color=B, alpha=.45, label="SegFormer-B2 (ours)")
        hi = max(d[gs].max(), d[f"{k}_m2f"].max(), d[f"{k}_segf"].max()) * 1.03
        ax.plot([0, hi], [0, hi], color=MUTED, lw=.8, ls="--")
        ax.set_xlim(0, hi), ax.set_ylim(0, hi)
        ax.set_title(f"{lab}\nCCC {ag[k]['gs_vs_m2f']['ccc']:.3f} / {ag[k]['gs_vs_segf']['ccc']:.3f}", fontsize=9)
        ax.set_xlabel("Global Streetscapes (provided)")
    axs[0].set_ylabel("Our re-segmentation")
    axs[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "agreement.png", dpi=150)


def density_fig():
    w = pd.read_csv("outputs/window_cells.csv")
    isl = pd.read_csv("outputs/island_cells.csv")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for ax, d, y, t in [(axs[0], w, "green_view_index", "A. GS window (110 cells, provided GVI)"),
                        (axs[1], isl, "gvi", "B. Island-wide (our Mask2Former GVI)")]:
        sc = ax.scatter(np.log1p(d.gfa_intensity), d[y], c=d.park_share, cmap="Greens", s=10, vmin=0, vmax=1,
                        edgecolors=MUTED, linewidths=.2)
        ax.set_xlabel("log(1 + GFA intensity)  [URA MP2019 GPR x parcel area / cell area]")
        ax.set_title(t, fontsize=9)
    axs[0].set_ylabel("Green view index (250 m cell mean)")
    cb = fig.colorbar(sc, ax=axs, shrink=.8)
    cb.set_label("park / open-space share of cell")
    fig.savefig(FIG / "density_gvi.png", dpi=150, bbox_inches="tight")


def power_fig():
    p = json.loads(Path("outputs/power.json").read_text())
    g = pd.DataFrame(p["grid"])
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for eff, col in zip(sorted(g.effect.unique()), ["#9aa093", "#7fb59a", A, "#173b2a"]):
        s = g[g.effect == eff]
        ax.plot(s.raters, s.power, marker="o", color=col, label=f"beta = {eff}")
    ax.axhline(.8, color=MUTED, lw=.8, ls="--")
    ax.set_xscale("log"), ax.set_xticks(sorted(g.raters.unique()), [str(x) for x in sorted(g.raters.unique())])
    ax.set_xlabel("respondents (20 pairs each)"), ax.set_ylabel("simulated power")
    ax.set_title(f"{p['n_stimuli']} scenes, scene-level SD {p['item_sd']}", fontsize=9)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "power.png", dpi=150)


def geojson():
    isl = pd.read_csv("outputs/island_cells.csv")
    soft = set(pd.read_csv("outputs/softened_candidates.csv").cell)
    tr = Transformer.from_crs(3414, 4326, always_xy=True)
    lon, lat = tr.transform(isl.cx.values, isl.cy.values)
    feats = []
    for r, x, y in zip(isl.itertuples(), lon, lat):
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(x, 5), round(y, 5)]},
                      "properties": {"cell": r.cell, "pa": str(r.PLN_AREA_N).title(), "gvi": round(r.gvi, 3),
                                     "svi": round(r.svi, 3), "bld": round(r.building_share, 3),
                                     "gfa": round(r.gfa_intensity, 2), "soft": r.cell in soft}})
    Path("docs/data").mkdir(parents=True, exist_ok=True)
    Path("docs/data/island_cells.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "license": "CC BY-SA 4.0 (derived from KartaView imagery; density from "
         "URA/HDB data under the Singapore Open Data Licence v1.0)", "features": feats}))


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    agreement_fig()
    density_fig()
    power_fig()
    geojson()
    print("figures written")
