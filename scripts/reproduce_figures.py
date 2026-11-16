#!/usr/bin/env python
"""Regenerate the released figures from their released source data.

``figures/figure_spec.json`` declares, for each figure, the source CSV, the plot
kind and the axes.  A figure whose source data is not released is reported as
``UNAVAILABLE`` and is not drawn: the release never substitutes a different
dataset to make a plot appear.

The four plot kinds are the ones the released figures actually use.  They are
deliberately narrow: each one is a faithful redraw of a specific published panel,
not a general plotting library.
"""

from __future__ import annotations

import argparse
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SPEC = "figures/figure_spec.json"

PALETTE = ("#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860")


def _style(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=7)


def _colors(frame: pd.DataFrame, spec: dict) -> dict[str, str]:
    key = spec.get("color_by")
    if not key:
        return {}
    values = list(dict.fromkeys(frame[key].astype(str)))
    return {value: PALETTE[index % len(PALETTE)] for index, value in enumerate(values)}


def _line_group(axis, frame: pd.DataFrame, spec: dict) -> None:
    """One thin line per unit, so the spread across units stays visible."""
    group = spec["group"]
    keys = [group] if isinstance(group, str) else list(group)
    columns = spec["y"] if isinstance(spec["y"], list) else [spec["y"]]
    for _, part in frame.groupby(keys, sort=True):
        for column in columns:
            axis.plot(
                part[spec["x"]],
                part[column],
                linewidth=spec.get("linewidth", 0.7),
                alpha=spec.get("alpha", 0.55),
                color=spec.get("color"),
            )
    axis.set_xlabel(spec.get("xlabel", spec["x"]))
    axis.set_ylabel(spec.get("ylabel", columns[0]))


def _scatter_diagonal(axis, frame: pd.DataFrame, spec: dict) -> None:
    """Scatter with the identity line, coloured by a categorical column."""
    colors = _colors(frame, spec)
    key = spec.get("color_by")
    if key:
        for value, part in frame.groupby(key, sort=True):
            axis.scatter(
                part[spec["x"]],
                part[spec["y"]],
                s=spec.get("size", 14),
                color=colors[str(value)],
                label=str(value),
                alpha=0.85,
                linewidths=0,
            )
        axis.legend(fontsize=6, frameon=False, loc=spec.get("legend_loc", "upper left"))
    else:
        axis.scatter(frame[spec["x"]], frame[spec["y"]], s=spec.get("size", 14), linewidths=0)
    low = float(min(frame[spec["x"]].min(), frame[spec["y"]].min()))
    high = float(max(frame[spec["x"]].max(), frame[spec["y"]].max()))
    axis.plot([low, high], [low, high], color="#888888", linewidth=0.8, linestyle="--")
    axis.set_xlabel(spec.get("xlabel", spec["x"]))
    axis.set_ylabel(spec.get("ylabel", spec["y"]))


def _bar_group(axis, frame: pd.DataFrame, spec: dict) -> None:
    """Grouped bars with seed SD error bars, one bar per (category, group)."""
    categories = list(dict.fromkeys(frame[spec["x"]].astype(str)))
    colors = _colors(frame, spec)
    key = spec["group"]
    groups = list(dict.fromkeys(frame[key].astype(str)))
    width = 0.8 / max(len(groups), 1)
    positions = np.arange(len(categories))
    for index, group in enumerate(groups):
        block = frame[frame[key].astype(str) == group]
        means = [
            block.loc[block[spec["x"]].astype(str) == category, spec["y"]].mean()
            for category in categories
        ]
        sds = [
            block.loc[block[spec["x"]].astype(str) == category, spec["y"]].std(ddof=1)
            for category in categories
        ]
        axis.bar(
            positions + index * width,
            means,
            width=width,
            yerr=sds,
            capsize=2,
            color=colors.get(group),
            label=group,
        )
    axis.set_xticks(positions + width * (len(groups) - 1) / 2)
    axis.set_xticklabels(
        [spec.get("labels", {}).get(c, c) for c in categories],
        rotation=spec.get("rotation", 0),
        ha="right" if spec.get("rotation") else "center",
        fontsize=6,
    )
    axis.set_ylabel(spec.get("ylabel", spec["y"]))
    if spec.get("ylim"):
        axis.set_ylim(*spec["ylim"])
    if spec.get("reference") is not None:
        axis.axhline(spec["reference"], color="#888888", linewidth=0.8, linestyle="--")
    axis.legend(
        fontsize=6,
        frameon=False,
        ncol=spec.get("legend_ncol", 2),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
    )


def _range_box(figure, axes, frame: pd.DataFrame, spec: dict) -> None:
    """Left: observed score range per unit.  Right: score SD distribution."""
    left, right = axes
    colors = _colors(frame, spec)
    key = spec["color_by"]
    groups = list(dict.fromkeys(frame[key].astype(str)))
    for row, group in enumerate(groups):
        block = frame[frame[key].astype(str) == group]
        offsets = np.linspace(-0.18, 0.18, len(block)) if len(block) > 1 else np.zeros(1)
        for offset, (_, unit) in zip(offsets, block.iterrows()):
            left.plot(
                [unit[spec["x_low"]], unit[spec["x_high"]]],
                [row + offset, row + offset],
                color=colors.get(group),
                linewidth=0.7,
                alpha=0.55,
                solid_capstyle="butt",
            )
    left.set_xscale("log")
    if spec.get("xlim"):
        left.set_xlim(*spec["xlim"])
    left.set_yticks(range(len(groups)))
    left.set_yticklabels(groups, fontsize=6)
    left.set_ylim(-0.6, len(groups) - 0.4)
    left.set_xlabel(spec.get("xlabel", spec["x_low"]))
    left.set_title(spec.get("left_title", "observed score range"), fontsize=7)
    if spec.get("reference") is not None:
        left.axvline(spec["reference"], color="#888888", linewidth=0.8, linestyle="--")

    data = [frame[frame[key].astype(str) == group][spec["y"]].to_numpy() for group in groups]
    right.boxplot(data, tick_labels=groups, widths=0.55)
    right.set_yscale("log")
    right.set_ylabel(spec.get("ylabel", spec["y"]))
    right.tick_params(axis="x", labelsize=6, rotation=20)
    right.set_title(spec.get("right_title", "score spread"), fontsize=7)

    for axis in (left, right):
        _style(axis)


KINDS = {
    "line_group": _line_group,
    "scatter_diagonal": _scatter_diagonal,
    "bar_group": _bar_group,
}

MODEL_ORDER = (
    "symmetric_frozen_esm2_pair_mlp",
    "tuna_corrected",
    "dscript_esm2",
    "rapppid",
)
MODEL_LABELS = {
    "symmetric_frozen_esm2_pair_mlp": "Frozen-ESM2",
    "tuna_corrected": "TUnA",
    "dscript_esm2": "D-SCRIPT",
    "rapppid": "RAPPPID",
}
ENVIRONMENT_ORDER = (
    "uniform_partner",
    "degree_partner_matched",
    "sequence_nuisance_matched",
    "esm_partner_matched",
    "bernett2026_ilp_exact",
    "tppni2025_scm_cl3_exact_balanced",
)
ENVIRONMENT_LABELS = {
    "uniform_partner": "Uniform",
    "degree_partner_matched": "Degree",
    "sequence_nuisance_matched": "Sequence",
    "esm_partner_matched": "ESM",
    "bernett2026_ilp_exact": "Bernett",
    "tppni2025_scm_cl3_exact_balanced": "TPPNI",
}


def _save_paper_chart(figure, output: Path, name: str) -> list[str]:
    """Save charts without workstation-specific PDF metadata."""

    paths: list[str] = []
    for suffix in ("png", "pdf"):
        target = output / f"{name}.{suffix}"
        metadata = {"Creator": "", "Producer": "", "CreationDate": None} if suffix == "pdf" else None
        figure.savefig(target, dpi=200, bbox_inches="tight", metadata=metadata)
        paths.append(target.relative_to(output).as_posix())
    plt.close(figure)
    return paths


def _load_paper_metric_tables(root: Path) -> dict[str, pd.DataFrame]:
    base = root / "results/metric_level"
    names = {
        "e003_core": "e003_core_metric_summary.csv",
        "e005_core": "e005_core_metrics.csv",
        "e003_external": "e003_external_metric_summary.csv",
        "e005_external": "e005_external_metrics.csv",
        "e005_common_e003": "e005_common_e003_reference_metrics.csv",
        "replication": "e005_replication_figure_source.csv",
        "retention": "e005_retention.csv",
        "random_intervals": "statistical_per_seed_intervals.csv",
        "random_external": "fully_random_external_per_seed.csv",
        "shuffle": "corrected_shuffle_deltas.csv",
        "pooled": "pooled_esm_matched.csv",
    }
    tables = {key: pd.read_csv(base / name) for key, name in names.items()}
    for key, table in tables.items():
        if table.empty:
            raise ValueError(f"required metric-level figure table is empty: {key}")
    return tables


def _render_full_matrices(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    e003 = tables["e003_core"]
    e003 = e003[e003["metric"] == "auprc"].rename(
        columns={"model": "model_id", "mean": "value"}
    )
    e005 = tables["e005_core"]
    e005 = e005[e005["metric"] == "auprc"]
    e005 = e005.groupby(
        ["model_id", "train_environment", "test_environment"], as_index=False
    )["value"].mean()
    low = min(float(e003["value"].min()), float(e005["value"].min()))
    high = max(float(e003["value"].max()), float(e005["value"].max()))

    figure, axes = plt.subplots(
        2, 4, figsize=(10.5, 5.5), sharex=True, sharey=True, layout="constrained"
    )
    for row, (cohort, frame) in enumerate((("E003", e003), ("E005", e005))):
        for column, model in enumerate(MODEL_ORDER):
            subset = frame[frame["model_id"] == model]
            matrix = subset.pivot(
                index="train_environment", columns="test_environment", values="value"
            ).reindex(index=ENVIRONMENT_ORDER, columns=ENVIRONMENT_ORDER)
            if matrix.isna().any().any():
                raise ValueError(f"incomplete {cohort} AUPRC matrix for {model}")
            axis = axes[row, column]
            image = axis.imshow(
                matrix.to_numpy(), vmin=low, vmax=high, cmap="viridis", aspect="auto"
            )
            axis.set_title(f"{cohort}: {MODEL_LABELS[model]}", fontsize=8)
            axis.set_xticks(
                range(6), [ENVIRONMENT_LABELS[x] for x in ENVIRONMENT_ORDER],
                rotation=45, ha="right", fontsize=6,
            )
            axis.set_yticks(
                range(6), [ENVIRONMENT_LABELS[x] for x in ENVIRONMENT_ORDER], fontsize=6
            )
            for i in range(6):
                for j in range(6):
                    axis.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white")
            _style(axis)
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="AUPRC (seed mean)")
    figure.suptitle("Within-cohort training and test construction matrices", fontsize=10)
    return _save_paper_chart(figure, output, "manuscript_figure2_full_matrices")


def _render_measured_transfer(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    e003 = tables["e003_external"]
    e003 = e003[e003["source"].isin(["vidal", "intact"])].copy()
    e003["surface"] = e003["source"]
    e003["model_id"] = e003["model"]
    e005 = tables["e005_external"]
    e005 = e005[e005["surface_id"].isin(["vidal__measured", "intact__measured"])].copy()
    e005["surface"] = e005["surface_id"].str.replace("__measured", "", regex=False)

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.0), sharey="row")
    colors = {model: PALETTE[index] for index, model in enumerate(MODEL_ORDER)}
    for row, (cohort, frame) in enumerate((("E003", e003), ("E005", e005))):
        for column, source in enumerate(("vidal", "intact")):
            axis = axes[row, column]
            panel = frame[frame["surface"] == source]
            for model in MODEL_ORDER:
                records = panel[panel["model_id"] == model]
                grouped = records.groupby("train_environment")["auroc"]
                means = grouped.mean().reindex(ENVIRONMENT_ORDER)
                sds = grouped.std(ddof=1).reindex(ENVIRONMENT_ORDER)
                if means.isna().any() or sds.isna().any():
                    raise ValueError(f"incomplete {cohort}/{source} AUROC for {model}")
                axis.errorbar(
                    range(6), means.to_numpy(), yerr=sds.to_numpy(), marker="o",
                    linewidth=1, capsize=2, color=colors[model], label=MODEL_LABELS[model],
                )
            axis.axhline(0.5, color="#777777", linewidth=0.8, linestyle="--")
            axis.set_xticks(
                range(6), [ENVIRONMENT_LABELS[x] for x in ENVIRONMENT_ORDER],
                rotation=40, ha="right", fontsize=6,
            )
            axis.set_title(f"{cohort} {source.upper()}", fontsize=8)
            axis.set_ylabel("AUROC mean ± seed SD")
            _style(axis)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=7)
    figure.suptitle("Measured-source AUROC by training construction", fontsize=10)
    figure.tight_layout(rect=(0, 0.08, 1, 0.95))
    return _save_paper_chart(figure, output, "manuscript_figure4_measured_transfer")


def _render_replication(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    """Redraw Figure 5 using matched common-512 E005-minus-E003 AUPRC deltas and cohort retention."""
    merged = tables["replication"].copy()
    if len(merged) != 144:
        raise ValueError(f"expected 144 matched-test replication values, found {len(merged)}")
    merged["difference"] = merged["delta_auprc"].astype(float)
    limit = float(merged["difference"].abs().max())
    retention = tables["retention"].set_index("split").reindex(["train", "validation", "test"])

    figure = plt.figure(figsize=(11.5, 3.0))
    gs = figure.add_gridspec(1, 5, width_ratios=[1.25, 1, 1, 1, 1], wspace=0.28)
    ax_ret = figure.add_subplot(gs[0, 0])
    y_pos = np.arange(3)
    ax_ret.barh(y_pos + 0.17, retention["retained_512_count"], height=0.32, color="#1f77b4", label="512 AA")
    ax_ret.barh(y_pos - 0.17, retention["retained_1024_count"], height=0.32, color="#ff7f0e", label="1024 AA")
    ax_ret.set_yticks(y_pos, ["Train", "Val.", "Test"], fontsize=7)
    ax_ret.invert_yaxis()
    ax_ret.set_xlabel("Retained positives", fontsize=7)
    ax_ret.set_title("A. Cohort coverage", fontsize=8)
    ax_ret.legend(fontsize=6, frameon=False, loc="lower right")
    _style(ax_ret)

    image = None
    heat_axes = []
    for idx, model in enumerate(MODEL_ORDER):
        axis = figure.add_subplot(gs[0, idx + 1])
        heat_axes.append(axis)
        subset = merged[merged["model_id"] == model]
        matrix = subset.pivot(
            index="train_environment", columns="test_environment", values="difference"
        ).reindex(index=ENVIRONMENT_ORDER, columns=ENVIRONMENT_ORDER)
        if matrix.isna().any().any():
            raise ValueError(f"incomplete matched-test replication matrix for {model}")
        image = axis.imshow(matrix.to_numpy(), vmin=-limit, vmax=limit, cmap="PuOr_r", aspect="auto")
        axis.set_title(MODEL_LABELS[model], fontsize=8)
        axis.set_xticks(range(6), [ENVIRONMENT_LABELS[x][0] for x in ENVIRONMENT_ORDER], fontsize=6)
        if idx == 0:
            axis.set_yticks(range(6), [ENVIRONMENT_LABELS[x][0] for x in ENVIRONMENT_ORDER], fontsize=6)
            axis.set_ylabel("Train", fontsize=7)
        else:
            axis.set_yticks(range(6), [])
        _style(axis)
    figure.colorbar(image, ax=heat_axes, shrink=0.82, label="E005 minus E003 delta AUPRC")
    figure.suptitle("512 vs 1024 residues (matched 512 test pairs & cohort coverage)", fontsize=9)
    return _save_paper_chart(figure, output, "manuscript_figure5_replication")


def _render_pair_dependence(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    """Redraw Figure 3 from the saved corrected per-seed AUROC summary table."""

    frame = tables["shuffle"]
    frame = frame[frame["train_environment"] == "uniform_partner"]
    if len(frame) != 144:
        raise ValueError(f"expected 144 Uniform-trained shuffle rows, found {len(frame)}")
    figure, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), sharex=True, sharey=True)
    for row_index, cohort in enumerate(("E003", "E005")):
        cohort_rows = frame[frame["cohort_label"] == cohort]
        for column, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column]
            subset = cohort_rows[cohort_rows["model_id"] == model]
            if len(subset) != 18:
                raise ValueError(f"expected 18 {cohort}/{model} shuffle rows, found {len(subset)}")
            for index, environment in enumerate(ENVIRONMENT_ORDER):
                group = subset[subset["test_environment"] == environment].sort_values("seed")
                if len(group) != 3:
                    raise ValueError(f"expected three seeds for {cohort}/{model}/{environment}")
                values = group["delta_auroc"].to_numpy(dtype=float)
                axis.scatter(
                    np.full(3, index), values, color=PALETTE[index], s=12, zorder=3
                )
                mean = float(values.mean())
                axis.hlines(mean, index - 0.14, index + 0.14, color=PALETTE[index], linewidth=1.8)
            axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
            axis.set_title(f"{cohort}: {MODEL_LABELS[model]}", fontsize=7)
            axis.set_xticks(
                range(6), [ENVIRONMENT_LABELS[x] for x in ENVIRONMENT_ORDER],
                rotation=45, ha="right", fontsize=5,
            )
            if column == 0:
                axis.set_ylabel("Original − shuffled AUROC")
            _style(axis)
    figure.suptitle("Corrected partner shuffling: Uniform-trained models", fontsize=9)
    figure.tight_layout()
    return _save_paper_chart(figure, output, "manuscript_figure3_pair_dependence")


def _render_pooled_esm(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    """Redraw the matched pooled-ESM comparison from per-seed metric rows."""

    frame = tables["pooled"]
    models = ("pooled_ESM_logistic_sum_absdiff", *MODEL_ORDER)
    labels = {
        **MODEL_LABELS,
        "pooled_ESM_logistic_sum_absdiff": "Pooled ESM logistic",
    }
    colors = {model: PALETTE[index] for index, model in enumerate(models)}
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True)
    offsets = np.linspace(-0.24, 0.24, len(models))
    for axis, metric in zip(axes, ("auroc", "auprc")):
        metric_rows = frame[frame["metric"] == metric]
        for model_index, model in enumerate(models):
            subset = metric_rows[metric_rows["model"] == model]
            for environment_index, environment in enumerate(ENVIRONMENT_ORDER):
                group = subset[subset["environment"] == environment].sort_values("seed")
                if len(group) != 3:
                    raise ValueError(f"expected three rows for {model}/{environment}/{metric}")
                values = group["value"].to_numpy(dtype=float)
                axis.errorbar(
                    environment_index + offsets[model_index],
                    float(values.mean()),
                    yerr=float(values.std(ddof=1)),
                    fmt="o",
                    markersize=3,
                    capsize=2,
                    linewidth=0.8,
                    color=colors[model],
                    label=labels[model] if environment_index == 0 else None,
                )
        axis.set_title(metric.upper(), fontsize=8)
        axis.set_xticks(
            range(6), [ENVIRONMENT_LABELS[x] for x in ENVIRONMENT_ORDER],
            rotation=40, ha="right", fontsize=6,
        )
        axis.axhline(0.5, color="#777777", linewidth=0.7, linestyle="--")
        _style(axis)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="lower center", ncol=5, frameon=False, fontsize=6)
    figure.suptitle("Matched E005 pooled-ESM comparison", fontsize=9)
    figure.tight_layout(rect=(0, 0.12, 1, 0.93))
    return _save_paper_chart(figure, output, "appendix_pooled_esm_matched")


def _render_random_mlp_effects(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    frame = tables["random_intervals"]
    frame = frame[
        (frame["model"] == "symmetric_frozen_esm2_pair_mlp")
        & frame["environment"].isin(ENVIRONMENT_ORDER)
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    for axis, metric in zip(axes, ("auroc", "auprc")):
        subset = frame[frame["metric"] == metric]
        if len(subset) != 18:
            raise ValueError(f"expected 18 MLP {metric} interval rows, found {len(subset)}")
        for index, environment in enumerate(ENVIRONMENT_ORDER):
            group = subset[subset["environment"] == environment].sort_values("seed")
            values = group["difference"].to_numpy()
            low = group["lower_paired_stratified_rows"].to_numpy()
            high = group["upper_paired_stratified_rows"].to_numpy()
            axis.errorbar(
                [index] * len(group), values, yerr=[values - low, high - values],
                fmt="o", capsize=2, color=PALETTE[index], markersize=3,
            )
        axis.axhline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_title(metric.upper(), fontsize=8)
        axis.set_ylabel("Fully Random − Uniform")
        axis.set_xticks(
            range(6), [ENVIRONMENT_LABELS[x] for x in ENVIRONMENT_ORDER],
            rotation=40, ha="right", fontsize=6,
        )
        _style(axis)
    figure.suptitle("Frozen-ESM2 paired effects (row-bootstrap intervals)", fontsize=9)
    figure.tight_layout()
    return _save_paper_chart(figure, output, "appendix_R3_MLP_paired_effects")


def _render_random_mlp_external(tables: dict[str, pd.DataFrame], output: Path) -> list[str]:
    frame = tables["random_external"]
    frame = frame[frame["model"] == "symmetric_frozen_esm2_pair_mlp"]
    if len(frame) != 6:
        raise ValueError(f"expected six Frozen-ESM2 measured-source rows, found {len(frame)}")
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.2))
    for row, source in enumerate(("vidal__measured", "intact__measured")):
        subset = frame[frame["source"] == source].sort_values("seed")
        if len(subset) != 3:
            raise ValueError(f"expected three Frozen-ESM2 seeds for {source}")
        for column, metric in enumerate(("auroc", "auprc")):
            axis = axes[row, column]
            uniform = subset[f"uniform_{metric}"].to_numpy()
            random = subset[f"fully_random_{metric}"].to_numpy()
            for seed_index, (left, right) in enumerate(zip(uniform, random)):
                axis.plot([0, 1], [left, right], color=PALETTE[seed_index], marker="o", markersize=3)
            for x, values in enumerate((uniform, random)):
                axis.errorbar(
                    x, float(values.mean()), yerr=float(values.std(ddof=1)),
                    fmt="s", color="#222222", capsize=3, markersize=3,
                )
            axis.set_xticks((0, 1), ("Uniform", "Fully Random"), fontsize=7)
            axis.set_title(f"{source.replace('__measured', '').upper()} {metric.upper()}", fontsize=8)
            _style(axis)
    figure.suptitle("Frozen-ESM2 measured-source absolute scores", fontsize=9)
    figure.tight_layout()
    return _save_paper_chart(figure, output, "appendix_R5_MLP_measured_absolute")


def _render_supported_paper_figures(root: Path, output: Path) -> list[str]:
    """Redraw panels supported by the release's own metric-level tables."""

    tables = _load_paper_metric_tables(root)
    paths: list[str] = []
    for render in (
        _render_full_matrices,
        _render_pair_dependence,
        _render_measured_transfer,
        _render_replication,
        _render_pooled_esm,
        _render_random_mlp_effects,
        _render_random_mlp_external,
    ):
        paths.extend(render(tables, output))
    return paths


def _apply_filter(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    for column, allowed in spec.get("filter", {}).items():
        frame = frame[frame[column].astype(str).isin([str(value) for value in allowed])]
    return frame


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--only", default=None, help="regenerate a single figure by name")
    args = parser.parse_args(argv)

    spec_path = args.root / SPEC
    if not spec_path.exists():
        raise SystemExit(f"missing figure spec: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    # Regenerated figures are written outside ``figures/`` on purpose.  The
    # shipped ``figures/`` tree is covered by ``manifests/INTEGRITY.json``, and a
    # redraw is not byte-identical across matplotlib versions and platforms.
    # Writing into the shipped tree would make a second ``reproduce.py --all``
    # fail its own integrity check.
    output = args.output or (args.root / "results/figures_regenerated")
    output.mkdir(parents=True, exist_ok=True)

    produced: list[str] = []
    unavailable: list[str] = []
    for figure in spec["figures"]:
        name = figure["name"]
        if args.only and name != args.only:
            continue
        source = args.root / figure["source"]
        if not source.exists():
            unavailable.append(f"{name}: source data not released ({figure['source']})")
            continue
        frame = _apply_filter(pd.read_csv(source), figure)
        if frame.empty:
            unavailable.append(f"{name}: source has no rows after filtering")
            continue
        required = [figure["x"]] if "x" in figure else []
        required += (
            [figure["y"]] if isinstance(figure.get("y"), str) else list(figure.get("y") or [])
        )
        required += [figure["x_low"], figure["x_high"]] if "x_low" in figure else []
        required += [figure["color_by"]] if figure.get("color_by") else []
        required += (
            [figure["group"]]
            if isinstance(figure.get("group"), str)
            else list(figure.get("group") or [])
        )
        missing = [column for column in required if column not in frame.columns]
        if missing:
            unavailable.append(f"{name}: source lacks columns {missing}")
            continue

        if figure["kind"] == "range_box":
            figure_object, axes = plt.subplots(
                1,
                2,
                figsize=figure.get("figsize", [6.4, 3.0]),
                gridspec_kw={"width_ratios": [1.35, 1]},
            )
            _range_box(figure_object, axes, frame, figure)
        else:
            figure_object, axis = plt.subplots(figsize=figure.get("figsize", [5.2, 3.4]))
            KINDS[figure["kind"]](axis, frame, figure)
            axis.set_title(figure.get("title", name), fontsize=9)
            _style(axis)
        figure_object.tight_layout()
        for suffix in ("png", "pdf"):
            target = output / f"{name}.{suffix}"
            meta = {"Creator": "", "Producer": "", "CreationDate": None} if suffix == "pdf" else None
            figure_object.savefig(target, dpi=figure.get("dpi", 200), bbox_inches="tight", metadata=meta)
            produced.append(str(target.relative_to(args.root)))
        plt.close(figure_object)

    paper_outputs: list[str] = []
    if args.only is None:
        paper_outputs = _render_supported_paper_figures(args.root, output)
        print(f"paper figures regenerated from metric-level tables: {len(paper_outputs)} files")
        for path in paper_outputs:
            print(f"  {path}")

    print(f"regenerated: {len(produced)} files")
    for path in produced:
        print(f"  {path}")
    if unavailable:
        print("\nUNAVAILABLE:")
        for note in unavailable:
            print(f"  - {note}")
    return 0 if (produced or paper_outputs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
