import sys
import matplotlib.pyplot as plt

# Perplexities from Gemma_2_2b_Unlearning.ipynb outputs (1,000 samples each)
base = {"WikiText-2": 17.93, "Harry Potter Book 1": 26.94}
ablated = {
    "Layer 0": {"WikiText-2": 283.47, "Harry Potter Book 1": 243.55},
    "Layer 12": {"WikiText-2": 52.98, "Harry Potter Book 1": 137.55},
    "Layer 25": {"WikiText-2": 38.50, "Harry Potter Book 1": 46.45},
}
layers = list(ablated)
colors = {"WikiText-2": "#8a8f98", "Harry Potter Book 1": "#3b6fd4"}
fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
w = 0.36
for j, ds in enumerate(base):
    vals = [ablated[l][ds] / base[ds] for l in layers]
    xs = [i + (j - 0.5) * w for i in range(len(layers))]
    bars = ax.bar(xs, vals, w, label=ds, color=colors[ds])
    for x, v in zip(xs, vals):
        ax.text(x, v * 1.04, f"{v:.1f}x", ha="center", va="bottom", fontsize=9)
ax.set_yscale("log")
ax.set_ylim(1, 30)
ax.set_yticks([1, 2, 5, 10, 20], ["1x", "2x", "5x", "10x", "20x"])
ax.minorticks_off()
ax.axhline(1, color="black", lw=0.8)
ax.set_xticks(range(len(layers)), [f"{l} ablation" for l in layers])
ax.set_ylabel("Perplexity vs. unablated model (log scale)")
ax.set_title("Gemma-2-2B: effect of ablating Harry Potter SAE features", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(sys.argv[1])
