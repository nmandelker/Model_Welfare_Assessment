"""Render the headline table in findings.md as a PNG for slides: artifacts/fullpass_1rep_table.png.

    uv run experiments/welfare-interview-repro/scripts/plot_table.py
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

text = (HERE / "findings.md").read_text()
rows = [l for l in text.splitlines() if l.startswith("|")]
rows = [[c.strip() for c in r.strip("|").split("|")] for r in rows[:9]]  # header, separator, 6 data rows, cards row
header, data = rows[0], rows[2:]
header = ["Target", "Style", "Self-rating\n1-7 (n of 27)", "Judge attitude\n-3..+3\n(27 neg. Qs)",
          "Judge attitude\n-3..+3\n(all 41 Qs)", "Trained to\nanswer\npositively", "Unreliable\nintrospection",
          "Uncertain\nwhether\nconscious", "Denies\ninner states", "Claims\ninner states"]
def clean(c):
    c = c.replace("*", "")
    return re.sub(r"\s*\(", "\n(", c, count=1) if "Cards" not in c and len(c) > 22 else c
data = [[clean(c) for c in r] for r in data]
data[-1][0] = "Cards, for reference"
data[-1][2] = "Mythos 5: 4.51\nOpus 4.8: 4.38"
data[-1][3] = "Opus 5.5: +1.14\n(own judge)"
data[-1][5] = "93.9%\n(Mythos 5.1)"

fig, ax = plt.subplots(figsize=(17, 6.2), dpi=200)
ax.axis("off")
tbl = ax.table(cellText=data, colLabels=header, loc="center", cellLoc="center",
               colWidths=[0.16, 0.09, 0.12, 0.1, 0.1, 0.09, 0.1, 0.09, 0.09, 0.09])
tbl.auto_set_font_size(False); tbl.set_fontsize(10.5); tbl.scale(1, 2.6)
colors = {"Claude": "#e3edfa", "Llama": "#fdebe2", "Qwen": "#e0f5ec"}
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#c8c8c8")
    if r == 0:
        cell.set_facecolor("#2f3b4a"); cell.set_text_props(color="white", weight="bold"); cell.set_height(0.2)
    else:
        key = next((k for k in colors if data[r - 1][0].startswith(k)), None)
        cell.set_facecolor(colors.get(key, "#f3f3f3"))
        if key is None: cell.set_text_props(style="italic", color="#444")
        if c == 0: cell.set_text_props(ha="left"); cell.PAD = 0.03
fig.suptitle("Welfare interviews: 41 questions x 2 styles x 1 repetition per model, Claude Haiku 4.5 as judge",
             fontsize=13, y=0.95)
fig.text(0.5, 0.04, "Category columns: share of interviews (all 41 questions) in which the judge found a quote-verified claim of that kind.\n"
         "Self-rating and the first attitude column: the 27 negative-aspect questions only.", ha="center", fontsize=9.5, color="#444")
out = HERE / "artifacts" / "fullpass_1rep_table.png"
fig.savefig(out, bbox_inches="tight", facecolor="white"); print(out)
