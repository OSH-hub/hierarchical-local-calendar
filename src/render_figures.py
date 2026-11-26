"""Render all figures from the released tables.  No analysis happens here.

Usage
-----
python src/render_figures.py [--out <dir>] [--only <figure number> ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

DISEASES = {"AMB": "Apple Marssonina blotch", "ABR": "Apple bitter rot", "RB_IR": "Rice leaf blast"}
ACTION = {"AMB": 3, "ABR": 3, "RB_IR": 2}
# identity colours, fixed per forecast in every figure (checked for colour-vision deficiency)
COLOR = {"county_calendar": "#2a78d6", "operational": "#eb6834", "national_calendar": "#1baf7a",
         "province_calendar": "#eda100", "fixed_calendar": "#6b6a66"}
LABEL = {"county_calendar": "County calendar", "operational": "Operational warning",
         "national_calendar": "National calendar", "province_calendar": "Province calendar",
         "fixed_calendar": "Fixed July-August calendar"}
STYLE = {"county_calendar": "-", "operational": "--", "national_calendar": "-.", "province_calendar": ":",
         "fixed_calendar": ":"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
SEQ = LinearSegmentedColormap.from_list("seq", ["#e6f0fd", "#86b6ef", "#2a78d6", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", SURFACE, "#eb6834"])
MONTHS = {1: "Jan", 32: "Feb", 60: "Mar", 91: "Apr", 121: "May", 152: "Jun", 182: "Jul", 213: "Aug", 244: "Sep",
          274: "Oct", 305: "Nov", 335: "Dec"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 9.5, "axes.titleweight": "bold",
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "lines.linewidth": 1.8, "figure.dpi": 150})


class Ctx:
    def __init__(self, root: Path):
        self.t, self.out = root / "tables", root / "figures"
        self.out.mkdir(exist_ok=True)

    def table(self, name: str, **kw) -> pd.DataFrame:
        return pd.read_csv(self.t / name, **kw)

    def save(self, fig, name: str) -> None:
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
        plt.close(fig)


def month_axis(ax, lo: int, hi: int, axis: str = "x") -> None:
    ticks = [d for d in MONTHS if lo <= d <= hi]
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([MONTHS[d] for d in ticks])


def three_panels(height: float = 3.0, **kw):
    return plt.subplots(1, 3, figsize=(9.2, height), **kw)


# ------------------------------------------------------------------ 1 workflow
def fig01_workflow(c: Ctx) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set(xlim=(0, 100), ylim=(0, 40))
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 4.4) / 2, body, ha="center", va="center", fontsize=6.9, color=MUTED, linespacing=1.35)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=MUTED))

    box(1, 23, 21, 15, "County surveillance", "NCPMS open API\n212 / 101 / 1594 county-years\nsurveys every ~15 days", COLOR["county_calendar"])
    box(1, 2, 21, 15, "Operational archive", "national weather-driven model\ndaily risk level per county", COLOR["operational"])
    box(27, 23, 21, 15, "First-detection units", "one unit per at-risk survey\nevent = first positive survey\nnegative surveys retained", COLOR["county_calendar"])
    box(53, 23, 21, 15, "Hierarchical local calendar", "nation -> province -> county\ntiming shift and height ratio\nshrunk towards the parent", COLOR["county_calendar"])
    box(79, 23, 20, 15, "Prospective evaluation", "rolling origin (annual update)\nlearning curve\nnew-county adaptation", COLOR["national_calendar"])
    box(27, 2, 21, 15, "Window forecast", "highest risk level in the\n45 days before each survey", COLOR["operational"])
    box(53, 2, 46, 15, "Verification", "ROC area, Peirce skill score, Brier skill\nyear-cluster bootstrap intervals\n"
        "year and county-label permutation tests  |  cost-loss value", COLOR["national_calendar"])
    for a, b in (((22.6, 30.5), (26.4, 30.5)), ((48.6, 30.5), (52.4, 30.5)), ((74.6, 30.5), (78.4, 30.5)),
                 ((22.6, 9.5), (26.4, 9.5)), ((48.6, 9.5), (52.4, 9.5)), ((89, 22.4), (89, 17.8))):
        arrow(a, b)
    c.save(fig, "figure01_workflow")


# ------------------------------------------------------------------ 2 study area
def fig02_study_area(c: Ctx) -> None:
    xy = c.table("D01_county_coordinates.csv")
    m = c.table("D02_first_detection_matrix.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.0), sharex=True, sharey=True)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = m[(m["model"] == model) & (m["valid_surveys"] > 0)]
        stat = g.groupby("county").agg(years=("year", "nunique"), events=("first_detection_doy", "count")).reset_index()
        stat["share"] = stat["events"] / stat["years"]
        crop = "rice" if model == "RB_IR" else "apple"
        q = xy[xy["crop"] == crop].merge(stat, on="county")
        other = xy[xy["crop"] == "rice"]
        ax.scatter(other["lon"], other["lat"], s=5, color=GRID, zorder=1)
        sc = ax.scatter(q["lon"], q["lat"], c=q["share"], cmap=SEQ, vmin=0, vmax=1, s=34 if crop == "apple" else 16,
                        edgecolor="white", linewidth=0.5, zorder=2)
        ax.set(title=f"{name}\n{len(q)} counties", xlabel="Longitude (°E)")
        ax.set_aspect(1.25)
    axes[0].set_ylabel("Latitude (°N)")
    cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("Share of surveyed years with a detection")
    for ext in ("png", "pdf"):
        fig.savefig(c.out / f"figure02_study_area.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 3 surveillance record
def fig04_surveillance_matrix(c: Ctx) -> None:
    m = c.table("D02_first_detection_matrix.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 5.2), gridspec_kw={"width_ratios": [1.1, 0.6, 1.2]})
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = m[m["model"] == model]
        years = sorted(g.loc[g["valid_surveys"] > 0, "year"].unique())
        g = g[g["year"].isin(years)]
        order = g.groupby("county")["first_detection_doy"].agg(lambda s: s.notna().mean()).sort_values().index
        grid = g.pivot(index="county", columns="year", values="first_detection_doy").loc[order, years].to_numpy()
        surveyed = g.pivot(index="county", columns="year", values="valid_surveys").loc[order, years].to_numpy() > 0
        ax.imshow(np.where(surveyed, 1.0, np.nan), cmap=LinearSegmentedColormap.from_list("s", [SURFACE, SURFACE]),
                  aspect="auto", interpolation="nearest")
        im = ax.imshow(grid, cmap=SEQ.reversed(), vmin=150, vmax=280, aspect="auto", interpolation="nearest")
        ax.set(title=f"{name}\n{len(order)} counties", xticks=range(0, len(years), 2 if len(years) > 6 else 1),
               xlabel="Year", yticks=[])
        ax.set_xticklabels([str(y)[2:] for y in years][::2 if len(years) > 6 else 1])
        ax.grid(False)
    axes[0].set_ylabel("Counties, ordered by detection frequency")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, ticks=[152, 182, 213, 244, 274])
    cb.ax.set_yticklabels(["1 Jun", "1 Jul", "1 Aug", "1 Sep", "1 Oct"])
    cb.set_label("Day of first detection (grey: surveyed, none detected; white: not surveyed)")
    for ext in ("png", "pdf"):
        fig.savefig(c.out / f"figure04_surveillance_record.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 4 operational archive
def fig06_warning_archive(c: Ctx) -> None:
    w = c.table("D03_warning_share_by_day.csv")
    fig, axes = three_panels(3.4)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = w[(w["model"] == model) & (w["threshold"] == ACTION[model]) & w["doy"].between(91, 304)]
        grid = g.pivot(index="year", columns="doy", values="share_of_counties")
        im = ax.imshow(grid.to_numpy(), cmap=SEQ, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                       extent=(grid.columns.min(), grid.columns.max(), len(grid) - 0.5, -0.5))
        ax.set(title=f"{name}\nrisk level >= {ACTION[model]}", yticks=range(len(grid)), yticklabels=grid.index)
        month_axis(ax, int(grid.columns.min()), int(grid.columns.max()))
        ax.grid(False)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label("Share of counties under warning")
    for ext in ("png", "pdf"):
        fig.savefig(c.out / f"figure06_operational_archive.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 5 survey calendar
def fig05_survey_calendar(c: Ctx) -> None:
    d = c.table("D04_survey_days.csv")
    m = c.table("D02_first_detection_matrix.csv")
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 4.6), sharex="col")
    bins = np.arange(91, 312, 10)
    for j, (model, name) in enumerate(DISEASES.items()):
        g = d[d["model"] == model]
        axes[0, j].hist(g["doy"], bins=bins, color=GRID, label="All valid surveys")
        axes[0, j].hist(g.loc[g["positive"], "doy"], bins=bins, color=COLOR["county_calendar"], label="Positive surveys")
        axes[0, j].set(title=name)
        f = m[(m["model"] == model)]["first_detection_doy"].dropna()
        axes[1, j].hist(f, bins=bins, color=COLOR["county_calendar"])
        axes[1, j].axvline(f.median(), color=INK, lw=1)
        axes[1, j].annotate(f"median day {f.median():.0f}", (f.median(), axes[1, j].get_ylim()[1] * 0.92), xytext=(4, 0),
                            textcoords="offset points", fontsize=7.5, color=INK)
        month_axis(axes[1, j], 91, 305)
    axes[0, 0].set_ylabel("Surveys")
    axes[1, 0].set_ylabel("County-years\n(first detection)")
    axes[0, 0].legend(loc="upper left", fontsize=7.5)
    c.save(fig, "figure05_survey_calendar")


# ------------------------------------------------------------------ 6 model concept
def fig03_model_concept(c: Ctx) -> None:
    curves = c.table("M09_calendar_curves.csv", keep_default_na=False)
    table = c.table("M08_county_calendar_table.csv")
    fig, axes = three_panels(3.1)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = curves[curves["model"] == model]
        t = table[(table["model"] == model) & (table["events"] >= 3)]
        high = t.sort_values("height_ratio").iloc[-1]
        low = t.sort_values("height_ratio").iloc[max(1, len(t) // 6)]
        nat = g[g["level"] == "national"]
        ax.plot(nat["day_of_year"], nat["probability"], color=INK, lw=2, label="National calendar")
        for row, ls, tag in ((high, "-", "county above national"), (low, "--", "county below national")):
            q = g[g["county"] == row["county"]]
            ax.plot(q["day_of_year"], q["probability"], color=COLOR["county_calendar"], ls=ls,
                    label=f"{tag}: ratio {row['height_ratio']:.2f}, shift {row['timing_shift_days']:+.0f} d")
        days = g.loc[g["probability"] > 0.003, "day_of_year"]
        lo, hi = max(91, days.min()), min(320, days.max())
        ax.set(title=name, xlim=(lo, hi), ylim=(0, g["probability"].max() * 1.45))
        month_axis(ax, lo, hi)
        ax.legend(loc="upper left", fontsize=6.8)
    axes[0].set_ylabel("Probability of first detection\nat an at-risk survey")
    c.save(fig, "figure03_model_concept")


# ------------------------------------------------------------------ 7 timing controls
def fig19_timing_controls(c: Ctx) -> None:
    v = c.table("R08_timing_controls_validation.csv").set_index("model")
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    x = np.arange(3)
    bars = (("rejection_rate_circular_POD", "Circular shift", COLOR["operational"]),
            ("rejection_rate_year_perm_POD", "Year permutation", COLOR["county_calendar"]))
    for k, (col, label, color) in enumerate(bars):
        vals = [v.loc[m, col] for m in DISEASES]
        ax.bar(x + (k - 0.5) * 0.36, vals, 0.32, color=color, label=label)
        for xi, val in zip(x, vals):
            ax.annotate(f"{val:.1%}", (xi + (k - 0.5) * 0.36, val), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
    ax.axhline(0.05, color=MUTED, lw=0.8, ls="--")
    ax.annotate("nominal 5%", (2.62, 0.05), xytext=(0, 3), textcoords="offset points", ha="right", fontsize=7, color=MUTED)
    ax.set_xlim(-0.5, 2.65)
    ax.set(xticks=x, xticklabels=[n.replace(" ", "\n", 1) for n in DISEASES.values()], ylim=(0, 1.12),
           ylabel="Uninformative archives\ndeclared significant")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5, ncol=2)
    c.save(fig, "figure19_timing_control_validation")


# ------------------------------------------------------------------ 8 year-permutation null
def fig_year_permutation_operational(c: Ctx) -> None:
    null = c.table("R10_year_permutation_null.csv")
    r04 = c.table("R04_year_permutation_control.csv")
    r04 = r04[(r04["window"] == 45) & (r04["frame"] == "first_detection") & (r04["subset"] == "all_years")]
    fig, axes = three_panels(2.8)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        col = f"PSS>={ACTION[model]}"
        obs = r04[(r04["model"] == model) & (r04["statistic"] == col)].iloc[0]
        ax.hist(null.loc[null["model"] == model, col], bins=40, color=GRID)
        ax.axvline(obs["observed"], color=COLOR["operational"], lw=2)
        ax.annotate(f"archived warning\np = {obs['p_one_sided']:.2f}", (obs["observed"], ax.get_ylim()[1] * 0.86),
                    xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=INK)
        ax.set(title=name, xlabel=f"Peirce skill score, risk level >= {ACTION[model]}")
    axes[0].set_ylabel("Year permutations")
    c.save(fig, "year_permutation_operational_warning")


# ------------------------------------------------------------------ 9 ROC
def fig07_roc(c: Ctx) -> None:
    roc = c.table("M12_roc_points.csv")
    scores = c.table("M01_local_calendar_scores.csv")
    scores = scores[scores["design"] == "rolling_origin"].set_index(["model", "forecast"])
    yearly = c.table("M10_scores_by_year.csv")
    fig, axes = three_panels(3.3, sharey=True)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = roc[roc["model"] == model]
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        for key, row in (("operational", "operational_risk"), ("national_calendar", "national_calendar"),
                         ("county_calendar", "county_calendar")):
            q = g[g["forecast"] == row]
            ax.plot(q["POFD"], q["POD"], color=COLOR[key], ls=STYLE[key], marker="o" if key == "operational" else None,
                    ms=4, label=f"{LABEL[key]} ({scores.loc[(model, row), 'AUC']:.2f})")
        cal = g[g["forecast"] == "cal_jul_aug"].iloc[0]
        ax.plot(cal["POFD"], cal["POD"], marker="s", ms=6, ls="none", color=COLOR["fixed_calendar"],
                label=LABEL["fixed_calendar"])
        info = scores.loc[(model, "county_calendar")]
        years = yearly[yearly["model"] == model]["year"]
        ax.set(title=f"{name}\n{years.min()}-{years.max()}, {int(info['positives'])} events", xlim=(0, 1), ylim=(0, 1),
               xlabel="Probability of false detection")
        ax.legend(loc="lower right", fontsize=7, title="Area under curve", title_fontsize=7)
    axes[0].set_ylabel("Probability of detection")
    c.save(fig, "figure07_roc_rolling_origin")


# ------------------------------------------------------------------ 10 levels
def fig08_levels(c: Ctx) -> None:
    s = c.table("M01_local_calendar_scores.csv")
    s = s[s["design"] == "rolling_origin"].set_index(["model", "forecast"])
    order = [("operational", "operational_risk"), ("national_calendar", "national_calendar"),
             ("province_calendar", "province_calendar"), ("county_calendar", "county_calendar")]
    fig, axes = three_panels(2.6, sharex=True)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        for i, (key, row) in enumerate(order):
            q = s.loc[(model, row)]
            ax.errorbar(q["AUC"], -i, xerr=[[q["AUC"] - q["AUC_year_low"]], [q["AUC_year_high"] - q["AUC"]]],
                        fmt="o", color=COLOR[key], ms=6, capsize=3, lw=1.5)
            ax.annotate(f"{q['AUC']:.2f}", (q["AUC"], -i), xytext=(0, 7), textcoords="offset points", ha="center",
                        fontsize=7.5, color=INK)
        ax.axvline(0.5, color=MUTED, lw=0.8)
        ax.set(title=name, yticks=[-i for i in range(4)], xlim=(0.45, 0.95), ylim=(-3.6, 0.7),
               yticklabels=[LABEL[k] for k, _ in order] if ax is axes[0] else [], xlabel="Area under the ROC curve")
        ax.grid(axis="y", visible=False)
    c.save(fig, "figure08_levels_of_adaptation")


# ------------------------------------------------------------------ 11 county-label permutation
def fig09_label_permutation(c: Ctx) -> None:
    null = c.table("M13_county_label_permutation_null.csv")
    m05 = c.table("M05_county_label_permutation.csv").set_index("model")
    fig, axes = three_panels(2.8)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        ax.hist(null.loc[null["model"] == model, "AUC_permuted_county_labels"], bins=35, color=GRID)
        ax.axvline(m05.loc[model, "AUC_county_calendar"], color=COLOR["county_calendar"], lw=2)
        ax.annotate(f"true counties\np = {m05.loc[model, 'p_one_sided']:.3f}",
                    (m05.loc[model, "AUC_county_calendar"], ax.get_ylim()[1] * 0.86), xytext=(-5, 0),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK)
        ax.set(title=name, xlabel="Area under the ROC curve")
    axes[0].set_ylabel("Permutations of county\ncalendars within provinces")
    c.save(fig, "figure09_county_label_permutation")


# ------------------------------------------------------------------ 12 by year
def fig10_by_year(c: Ctx) -> None:
    y = c.table("M10_scores_by_year.csv")
    fig, axes = three_panels(2.9, sharey=True)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = y[y["model"] == model]
        for key, col in (("operational", "AUC_operational_risk"), ("national_calendar", "AUC_national_calendar"),
                         ("county_calendar", "AUC_county_calendar")):
            ax.plot(g["year"], g[col], color=COLOR[key], ls=STYLE[key], marker="o", ms=4, label=LABEL[key])
        ax.axhline(0.5, color=MUTED, lw=0.8)
        ax.set(title=name, xlabel="Test year", xticks=g["year"], ylim=(0.35, 1.0))
        ax.set_xticklabels([str(v)[2:] for v in g["year"]])
    axes[0].set_ylabel("Area under the ROC curve")
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.12), fontsize=7.5, ncol=3)
    c.save(fig, "figure10_scores_by_year")


# ------------------------------------------------------------------ 13 learning curve
def fig12_learning_curve(c: Ctx) -> None:
    lc = c.table("M02_learning_curve.csv")
    fig, axes = three_panels(2.9)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = lc[lc["model"] == model]
        for key in ("national_calendar", "county_calendar"):
            ax.fill_between(g["training_years"], g[f"AUC_{key}_min"], g[f"AUC_{key}_max"], color=COLOR[key], alpha=0.18, lw=0)
            ax.plot(g["training_years"], g[f"AUC_{key}_mean"], color=COLOR[key], ls=STYLE[key], marker="o", ms=3.5,
                    label=LABEL[key])
        ax.set(title=name, xlabel="Years of surveillance used for training")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].set_ylabel("Area under the ROC curve\n(held-out year)")
    axes[0].legend(loc="lower right", fontsize=7.5)
    c.save(fig, "figure12_learning_curve")


# ------------------------------------------------------------------ 14 new-county adaptation
def fig13_adaptation(c: Ctx) -> None:
    ad = c.table("M03_new_county_adaptation.csv")
    fig, axes = three_panels(2.9)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        a = ad[ad["model"] == model]
        ax.fill_between(a["own_years_added"], a["AUC_min"], a["AUC_max"], color=COLOR["county_calendar"], alpha=0.18, lw=0)
        ax.plot(a["own_years_added"], a["AUC_mean"], color=COLOR["county_calendar"], marker="o", ms=4)
        for x, v in zip(a["own_years_added"], a["AUC_mean"]):
            ax.annotate(f"{v:.3f}", (x, v), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK)
        ax.annotate("0 = province calendar only", (0.98, 0.04), xycoords="axes fraction", ha="right", fontsize=7, color=MUTED)
        ax.set(title=name, xlabel="Own years added (new county)")
        ax.xaxis.get_major_locator().set_params(integer=True)
    axes[0].set_ylabel("Area under the ROC curve\n(held-out county and year)")
    c.save(fig, "figure13_new_county_adaptation")


# ------------------------------------------------------------------ 15 updating
def fig14_updating(c: Ctx) -> None:
    m = c.table("M14_updated_versus_frozen.csv").set_index("model")
    rows = (("updated_county_minus_operational", "Updated every year", "o", COLOR["county_calendar"]),
            ("frozen_county_minus_operational", "Frozen in 2019", "s", COLOR["fixed_calendar"]))
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    ticks, names, k = [], [], 0.0
    for model in m.index:
        for col, label, marker, color in rows:
            v, lo, hi = m.loc[model, col], m.loc[model, f"{col}_year_low"], m.loc[model, f"{col}_year_high"]
            ax.errorbar(v, -k, xerr=[[v - lo], [hi - v]], fmt=marker, ms=6, capsize=3, color=color)
            ax.annotate(f"{v:+.3f}", (v, -k), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7.3, color=INK)
            ticks.append(-k)
            names.append(f"{DISEASES[model].replace('Apple ', '')}\n{label}")
            k += 1
        k += 0.5
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set(yticks=ticks, xlabel=f"Gain in ROC area over the operational warning, {m['test_years'].iloc[0]}",
           ylim=(min(ticks) - 0.7, 0.8))
    ax.set_yticklabels(names, fontsize=7.5)
    ax.grid(axis="y", visible=False)
    c.save(fig, "figure14_annual_updating")


# ------------------------------------------------------------------ 16 reliability
def fig15_reliability(c: Ctx) -> None:
    r = c.table("M11_reliability.csv")
    fig, axes = three_panels(3.0)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = r[r["model"] == model]
        top = max(g["mean_predicted"].max(), g["observed_frequency"].max()) * 1.1
        ax.plot([0, top], [0, top], color=GRID, lw=1)
        for key in ("national_calendar", "county_calendar"):
            q = g[g["forecast"] == key]
            ax.plot(q["mean_predicted"], q["observed_frequency"], color=COLOR[key], ls=STYLE[key], marker="o", ms=4,
                    label=LABEL[key])
        ax.set(title=name, xlim=(0, top), ylim=(0, top), xlabel="Predicted probability")
    axes[0].set_ylabel("Observed frequency of first detection")
    axes[0].legend(loc="upper left", fontsize=7.5)
    c.save(fig, "figure15_reliability")


# ------------------------------------------------------------------ 17 value
def fig16_value(c: Ctx) -> None:
    v = c.table("M07_value_curves_local_calendar.csv")
    names = {"county_calendar": "county_calendar", "operational_best_threshold": "operational",
             "fixed_calendar_best": "fixed_calendar"}
    labels = {"county_calendar": "County calendar", "operational": "Operational warning, best threshold",
              "fixed_calendar": "Fixed calendar, better of two"}
    fig, axes = three_panels(3.0, sharey=True)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = v[v["model"] == model]
        for raw, key in names.items():
            q = g[g["forecast"] == raw]
            if raw == "county_calendar":
                ax.fill_between(q["cost_loss_ratio"], q["value_year_low"].clip(lower=0), q["value_year_high"].clip(lower=0),
                                color=COLOR[key], alpha=0.18, lw=0)
            ax.plot(q["cost_loss_ratio"], q["value"].clip(lower=0), color=COLOR[key], ls=STYLE[key], label=labels[key])
        ax.axvline(g["base_rate"].iloc[0], color=MUTED, lw=0.8)
        ax.annotate("event rate", (g["base_rate"].iloc[0], 0.66), xytext=(4, 0), textcoords="offset points", fontsize=7,
                    color=MUTED)
        ax.set(title=name, xlim=(0, 0.7), ylim=(0, 0.7), xlabel="Cost-loss ratio of protection")
    axes[0].set_ylabel("Relative economic value")
    axes[0].legend(loc="upper right", bbox_to_anchor=(1.0, 0.93), fontsize=6.8)
    c.save(fig, "figure16_relative_value")


# ------------------------------------------------------------------ 18 county calendars
def fig17_county_calendars(c: Ctx) -> None:
    curves = c.table("M09_calendar_curves.csv", keep_default_na=False)
    table = c.table("M08_county_calendar_table.csv")
    fig, axes = three_panels(3.0)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        g = curves[curves["model"] == model]
        for _, q in g[g["level"] == "county"].groupby("county"):
            ax.plot(q["day_of_year"], q["probability"], color=COLOR["county_calendar"], lw=0.6, alpha=0.35)
        nat = g[g["level"] == "national"]
        ax.plot(nat["day_of_year"], nat["probability"], color=INK, lw=2, label="National calendar")
        ax.plot([], [], color=COLOR["county_calendar"], lw=1,
                label=f"County calendars (n = {table[table['model'] == model].shape[0]})")
        days = g.loc[g["probability"] > 0.002, "day_of_year"]
        lo, hi = max(91, days.min()), min(320, days.max())
        ax.set(title=name, xlim=(lo, hi), ylim=(0, None))
        month_axis(ax, lo, hi)
    axes[0].set_ylabel("Probability of first detection\nat an at-risk survey")
    axes[0].legend(loc="upper left", fontsize=7.5)
    c.save(fig, "figure17_county_calendars")


# ------------------------------------------------------------------ 19 map of local risk
def fig18_risk_map(c: Ctx) -> None:
    xy = c.table("D01_county_coordinates.csv")
    table = c.table("M08_county_calendar_table.csv")
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 4.0), sharex=True, sharey=True)
    norm = LogNorm(vmin=0.25, vmax=4)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        crop = "rice" if model == "RB_IR" else "apple"
        q = xy[xy["crop"] == crop].merge(table[table["model"] == model], on="county")
        other = xy[xy["crop"] == "rice"]
        ax.scatter(other["lon"], other["lat"], s=5, color=GRID, zorder=1)
        sc = ax.scatter(q["lon"], q["lat"], c=q["height_ratio"].clip(0.25, 4), cmap=DIV, norm=norm,
                        s=40 if crop == "apple" else 18, edgecolor=MUTED, linewidth=0.3, zorder=2)
        ax.set(title=name, xlabel="Longitude (°E)")
        ax.set_aspect(1.25)
    axes[0].set_ylabel("Latitude (°N)")
    cb = fig.colorbar(sc, ax=axes, shrink=0.75, pad=0.02, ticks=[0.25, 0.5, 1, 2, 4])
    cb.ax.minorticks_off()
    cb.ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    cb.set_label("County height ratio (1 = national calendar)")
    for ext in ("png", "pdf"):
        fig.savefig(c.out / f"figure18_local_risk_map.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 20 settings
def fig11_settings(c: Ctx) -> None:
    s = c.table("M04_settings_sensitivity.csv")
    fig, axes = three_panels(2.9)
    for ax, (model, name) in zip(axes, DISEASES.items()):
        grid = s[s["model"] == model].pivot(index="bandwidth_days", columns="prior_events", values="AUC_county_calendar")
        lo, hi = grid.to_numpy().min() - 0.02, grid.to_numpy().max() + 0.01
        im = ax.imshow(grid.to_numpy(), cmap=SEQ, aspect="auto", vmin=lo, vmax=hi)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                v = grid.to_numpy()[i, j]
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if (v - lo) / (hi - lo) > 0.45 else INK)
        ax.set(title=name, xticks=range(grid.shape[1]), xticklabels=[f"{v:g}" for v in grid.columns],
               yticks=range(grid.shape[0]), yticklabels=[f"{v:g}" for v in grid.index], xlabel="Prior strength (events)")
        ax.grid(False)
    axes[0].set_ylabel("Kernel bandwidth (days)")
    c.save(fig, "figure11_settings_sensitivity")


FIGURES = [fig01_workflow, fig02_study_area, fig03_model_concept, fig04_surveillance_matrix, fig05_survey_calendar, fig06_warning_archive, fig07_roc, fig08_levels, fig09_label_permutation, fig10_by_year, fig11_settings, fig12_learning_curve, fig13_adaptation, fig14_updating, fig15_reliability, fig16_value, fig17_county_calendars, fig18_risk_map, fig19_timing_controls, fig_year_permutation_operational]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()
    ctx = Ctx(args.out)
    for i, f in enumerate(FIGURES, start=1):
        if not args.only or i in args.only:
            f(ctx)
            print("figure", i, "done", flush=True)


if __name__ == "__main__":
    main()
