import sys
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Two rows: how features are chosen (top), how they are ablated at inference (bottom)
top = [
    ("Harry Potter\nBook 1", "#eef1f6"),
    ("Top 50\nkeywords", "#eef1f6"),
    ("NeuronPedia\nfeature search", "#eef1f6"),
    ("SAE feature IDs\nL0: 439  L12: 180\nL25: 200", "#dfe8fb"),
]
bottom = [
    ("Residual stream\nat layer L", "#eef1f6"),
    ("SAE encode", "#dfe8fb"),
    ("Zero the\nHarry Potter\nfeatures", "#3b6fd4"),
    ("SAE decode", "#dfe8fb"),
    ("Rest of\nGemma-2-2B", "#eef1f6"),
]

fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.2)
ax.axis("off")


def row(items, y, label):
    w, gap = 1.62, 0.38
    x0 = (10 - len(items) * w - (len(items) - 1) * gap) / 2
    centers = []
    for i, (text, color) in enumerate(items):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y - 0.5), w, 1.0, boxstyle="round,pad=0.02,rounding_size=0.08",
                                    fc=color, ec="#5b6270", lw=1))
        ax.text(x + w / 2, y, text, ha="center", va="center", fontsize=8.5,
                color="white" if color == "#3b6fd4" else "#1d2330")
        if i:
            ax.annotate("", (x, y), (x - gap, y), arrowprops=dict(arrowstyle="->", color="#5b6270"))
        centers.append(x + w / 2)
    ax.text(0.05, y + 0.72, label, fontsize=9.5, weight="bold", color="#1d2330")
    return centers


t = row(top, 3.2, "1. Find the concept's features")
b = row(bottom, 1.0, "2. Ablate them during the forward pass (one layer at a time)")
ax.annotate("", (b[2], 1.52), (t[3], 2.68),
            arrowprops=dict(arrowstyle="->", color="#3b6fd4", ls="--", connectionstyle="arc3,rad=0.15"))
fig.tight_layout()
fig.savefig(sys.argv[1])
