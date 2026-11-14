from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
s=json.loads((root/"records/summary.json").read_text(encoding="utf-8"))
fig,ax=plt.subplots(figsize=(7,7));ax.set_box_aspect(1)
for arm,label,color in zip(s["arms"],["0% synthetic (real only)","99% synthetic","100% synthetic"],["#178487","#bd7730","#8b2447"]):
 ax.plot(range(11),arm["perplexity_trajectory"],"o-",label=label,color=color,linewidth=2,markersize=4)
ax.set_yscale("log");ax.set_xticks(range(11));ax.set(xlabel="Training cycle",ylabel="Fixed real-test corpus perplexity (log scale)",title="SmolLM2: plain SGD, constant LR 0.01, greedy targets")
ax.grid(alpha=.2,which="both");ax.legend(loc="upper left");fig.tight_layout();fig.savefig(root/"test_perplexity.png",dpi=180);plt.close(fig)
