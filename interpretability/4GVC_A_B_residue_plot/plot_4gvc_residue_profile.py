#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.interpolate import PchipInterpolator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

DATA_CSV = Path("/mnt/data/4GVC_A_B_final_plot_data.csv")
META_JSON = Path("/mnt/data/4GVC_A_B_final_plot_metadata.json")
OUTPUT_STEM = Path("/mnt/data/4GVC_residue_prediction_profile")


def main():
    df = pd.read_csv(DATA_CSV).sort_values("local_index").reset_index(drop=True)
    metadata = json.loads(META_JSON.read_text(encoding="utf-8"))
    required = {"case_id", "local_index", "amino_acid", "true_label", "probability", "pred_label", "class"}
    assert required.issubset(df.columns), sorted(required - set(df.columns))
    assert len(df) == 94
    assert df["local_index"].tolist() == list(range(1, 95))
    assert np.isfinite(df["probability"]).all()
    assert df["probability"].between(0, 1).all()

    true_positions = df.loc[df["true_label"] == 1, "local_index"].tolist()
    pred_positions = df.loc[df["pred_label"] == 1, "local_index"].tolist()
    tp_positions = df.loc[df["class"] == "TP", "local_index"].tolist()
    fp_positions = df.loc[df["class"] == "FP", "local_index"].tolist()
    fn_positions = df.loc[df["class"] == "FN", "local_index"].tolist()

    assert true_positions == [21,22,23,24,25,26,27,28,29,40,43,72,76,79]
    assert pred_positions == [21,22,23,24,25,26,27,28,29,40,72,75,79]
    assert tp_positions == [21,22,23,24,25,26,27,28,29,40,72,79]
    assert fp_positions == [75]
    assert fn_positions == [43,76]
    assert metadata["pdb"] == "4GVC"
    assert metadata["protein_chain"] == "A"
    assert metadata["peptide_chain"] == "B"
    assert metadata["peptide"] == "TKQEEFYA"
    assert float(metadata["threshold"]) == 0.610

    x = df["local_index"].to_numpy()
    prob = df["probability"].to_numpy()
    threshold = float(metadata["threshold"])
    true_mask = df["true_label"].to_numpy(dtype=bool)
    x_smooth = np.linspace(x.min(), x.max(), 1200)
    prob_smooth = np.clip(PchipInterpolator(x, prob)(x_smooth), 0.0, 1.0)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })

    fig = plt.figure(figsize=(14.2, 6.6), facecolor="white")
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.62, 4.0, 1.0], hspace=0.055)
    ax_heat = fig.add_subplot(gs[0])
    ax_curve = fig.add_subplot(gs[1], sharex=ax_heat)
    ax_marks = fig.add_subplot(gs[2], sharex=ax_heat)


    image = ax_heat.imshow(
        prob[np.newaxis, :], aspect="auto", cmap="Reds", vmin=0, vmax=1,
        interpolation="bilinear", extent=[0.5, 94.5, 0, 1], origin="lower",
    )
    ax_heat.set_ylabel("Probability", labelpad=11)
    ax_heat.set_yticks([])
    ax_heat.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for side in ("top", "right", "bottom"):
        ax_heat.spines[side].set_visible(False)
    cax = inset_axes(ax_heat, width="0.8%", height="92%", loc="center right",
                     bbox_to_anchor=(0.022, 0, 1, 1), bbox_transform=ax_heat.transAxes, borderpad=0)
    colorbar = fig.colorbar(image, cax=cax, ticks=[0.0, 0.5, 1.0])
    colorbar.set_label("Binding probability", rotation=270, labelpad=12)
    colorbar.ax.tick_params(length=2.5, width=0.6)

    line_color = "#355C7D"
    ax_curve.fill_between(x_smooth, prob_smooth, 0, color="#355C7D", alpha=0.15, linewidth=0, zorder=1)
    ax_curve.plot(x_smooth, prob_smooth, color=line_color, linewidth=1.6, zorder=2)
    ax_curve.axhline(threshold, color="#8C8C8C", linewidth=1.2, linestyle="--",
                     zorder=1.5, label=f"Threshold = {threshold:.3f}")
    ax_curve.scatter(x[true_mask], prob[true_mask], s=34, facecolors="none",
                     edgecolors="#C44E52", linewidths=1.2, zorder=4,
                     label="True binding residue")
    ax_curve.set_ylim(0, 1.02)
    ax_curve.set_yticks(np.arange(0, 1.01, 0.2))
    ax_curve.set_ylabel("Probability")
    ax_curve.grid(axis="y", color="#D9D9D9", linestyle="--", linewidth=0.7, alpha=0.7)
    ax_curve.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.legend(loc="upper right", frameon=False, fontsize=10,
                    handlelength=2.0)

    ax_marks.scatter(true_positions, np.ones(len(true_positions)), marker="s", s=32,
                     color="#C44E52", edgecolors="none", zorder=3)
    ax_marks.scatter(pred_positions, np.zeros(len(pred_positions)), marker="^", s=38,
                     color="#355C7D", edgecolors="none", zorder=3)
    ax_marks.set_yticks([1, 0], labels=["True", "Predicted"])
    ax_marks.set_ylim(-0.65, 1.65)
    ax_marks.spines["top"].set_visible(False)
    ax_marks.spines["right"].set_visible(False)
    ax_marks.spines["left"].set_visible(False)
    ax_marks.tick_params(axis="y", length=0, pad=12)
    ax_marks.set_xlabel("Protein residue position", labelpad=8)

    for axis in (ax_heat, ax_curve, ax_marks):
        axis.set_xlim(0.5, 94.5)
    ax_marks.set_xticks([1, 20, 40, 60, 80, 94])

    fig.subplots_adjust(left=0.065, right=0.935, top=0.865, bottom=0.12)
    title = (f"{metadata['case_id'].upper()}  ·  peptide: {metadata['peptide']}  ·  "
             f"F1={float(metadata['f1']):.3f}, MCC={float(metadata['mcc']):.3f}")
    fig.text(0.066, 0.927, title, ha="left", va="center", fontsize=12.2, fontweight="semibold")

    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"True residues: {true_positions}")
    print(f"Predicted residues: {pred_positions}")
    print(f"TP: {tp_positions}")
    print(f"FP: {fp_positions}")
    print(f"FN: {fn_positions}")
    print(f"Threshold: {threshold:.3f}")
    print(df.loc[df["local_index"].isin([40, 43, 72, 75, 76, 79]),
                 ["local_index", "amino_acid", "true_label", "probability", "pred_label", "class"]].to_string(index=False))


if __name__ == "__main__":
    main()
