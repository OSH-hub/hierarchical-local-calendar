"""Tables 4 and 5 of the article, and the two rows added to Table 3, from the released tables.

Usage: python src/make_article_tables.py
Writes tables/A01_article_table4_data_efficiency.csv and
tables/A02_article_table5_practical_value.csv and prints the rows of Table 3.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = Path(__file__).resolve().parent.parent / "tables"
MODELS = {"AMB": "Marssonina blotch", "ABR": "Bitter rot", "RB_IR": "Rice leaf blast"}


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / name)


def pick(df: pd.DataFrame, **where):
    for key, value in where.items():
        df = df[df[key] == value]
    return df.iloc[0] if len(df) else None


def f3(x: float) -> str:
    return f"{x:.3f}"


def signed(x: float) -> str:
    return ("+" if x > 0 else "-") + f"{abs(x):.0f}"


def row(label: str, fn) -> list[str]:
    return [label] + [fn(m) for m in MODELS]


def main() -> None:
    m01 = read("M01_local_calendar_scores.csv")
    m01 = m01[m01["design"] == "rolling_origin"]
    m02, m03 = read("M02_learning_curve.csv"), read("M03_new_county_adaptation.csv")
    m05, m07 = read("M05_county_label_permutation.csv"), read("M07_value_curves_local_calendar.csv")
    m08, m14 = read("M08_county_calendar_table.csv"), read("M14_updated_versus_frozen.csv")

    def learning(model: str, years: int, column: str) -> str:
        r = pick(m02, model=model, training_years=years)
        return "" if r is None else f3(r[column])

    def learning_all(model: str) -> str:
        d = m02[m02["model"] == model]
        r = d.loc[d["training_years"].idxmax()]
        return f"{f3(r['AUC_county_calendar_mean'])} ({int(r['training_years'])} years)"

    def new_county(model: str, k: int) -> str:
        r = pick(m03, model=model, own_years_added=k)
        return "" if r is None else f3(r["AUC_mean"])

    def updating(model: str, column: str, interval: bool = False) -> str:
        r = pick(m14, model=model)
        if r is None:
            return ""
        if not interval:
            return f3(r[column])
        low = r[column + "_year_low"]
        low = 0.0 if abs(low) < 0.0005 else low
        return f"{f3(r[column])} ({f3(low)} to {f3(r[column + '_year_high'])})"

    def positive(model: str, forecast: str) -> str:
        return str(int((m07[(m07["model"] == model) & (m07["forecast"] == forecast)]["value"] > 0).sum()))

    def shift(model: str) -> str:
        d = m08[m08["model"] == model]["timing_shift_days"]
        return f"{signed(d.min())} to {signed(d.max())}"

    def height(model: str) -> str:
        d = m08[m08["model"] == model]["height_ratio"]
        return f"{d.min():.2f} to {d.max():.2f}"

    head = ["quantity"] + list(MODELS.values())
    table4 = [
        row("Learning curve: national calendar, 2 training years", lambda m: learning(m, 2, "AUC_national_calendar_mean")),
        row("Learning curve: county calendar, 2 training years", lambda m: learning(m, 2, "AUC_county_calendar_mean")),
        row("Learning curve: county calendar, 5 training years", lambda m: learning(m, 5, "AUC_county_calendar_mean")),
        row("Learning curve: county calendar, all training years", learning_all),
    ] + [row(f"New county: {k} own year{'' if k == 1 else 's'} added", lambda m, k=k: new_county(m, k)) for k in range(6)] + [
        row("Annual updating: updated every year", lambda m: updating(m, "AUC_updated_county")),
        row("Annual updating: frozen in 2019", lambda m: updating(m, "AUC_frozen_county")),
        row("Annual updating: updated minus frozen", lambda m: updating(m, "updated_county_minus_frozen_county", True)),
        row("Annual updating: updated minus operational warning", lambda m: updating(m, "updated_county_minus_operational")),
        row("Annual updating: frozen minus operational warning", lambda m: updating(m, "frozen_county_minus_operational")),
    ]
    table5 = [
        row("Brier skill relative to the national calendar",
            lambda m: f3(pick(m01, model=m, forecast="county_calendar")["Brier_skill_vs_national"])),
        row("Cost-loss ratios with positive value: county calendar", lambda m: positive(m, "county_calendar")),
        row("Cost-loss ratios with positive value: operational warning", lambda m: positive(m, "operational_best_threshold")),
        row("Cost-loss ratios with positive value: fixed calendar", lambda m: positive(m, "fixed_calendar_best")),
        row("County calendar table: counties", lambda m: str(int((m08["model"] == m).sum()))),
        row("County calendar table: timing shift, range (days)", shift),
        row("County calendar table: height ratio, range", height),
    ]
    pd.DataFrame(table4, columns=head).to_csv(TABLES / "A01_article_table4_data_efficiency.csv", index=False, lineterminator="\n")
    pd.DataFrame(table5, columns=head).to_csv(TABLES / "A02_article_table5_practical_value.csv", index=False, lineterminator="\n")
    print("Table 3, added rows")
    print(row("County calendar, height ratio only", lambda m: f3(pick(m01, model=m, forecast="county_height_only")["AUC"])))
    print(row("County-label permutation, one-sided p", lambda m: f"{pick(m05, model=m)['p_one_sided']:.3f}"))


if __name__ == "__main__":
    main()
