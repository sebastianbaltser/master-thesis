import csv
import pathlib
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.style.use("ggplot")

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

PATH = pathlib.Path(__file__).parent.absolute()

dates, spreads = [], [] 

with open(PATH / "jp_morgan_cva_fva.csv", "r") as file:
    reader = csv.reader(file)
    # Pop the header
    next(reader)
    for date, spread in reader:
        quarter = date[1]
        year = date[-2:]
        date = f"{quarter}Q{year}"

        dates.append(date)
        spreads.append(float(spread.replace(",", "")))


fig, ax = plt.subplots(figsize=(7, 3))

@ticker.FuncFormatter
def million_formatter(x, pos):
    sign = "-" if x < 0 else ""
    x_in_millions = abs(x * 1e-6)
    return f"{sign}\${x_in_millions:.0f}M"

ax.set_ylim([-1e9, 0.6e9])
ax.yaxis.set_major_formatter(million_formatter)

ax.set_title("JP Morgan Credit Adjustments \& Other")

ax.plot(dates, spreads, color=[i / 255 for i in [255, 68, 61]])
x_start, x_end = ax.get_xlim()
ax.plot([x_start, x_end], [0, 0], color="black", linestyle="--", linewidth=0.5)
ax.set_xlim(x_start, x_end)

for i, label in enumerate(ax.get_xticklabels()):
    if (i % 4) != 0:
        label.set_visible(False)

plt.tight_layout()
plt.savefig(PATH / "jp-morgan-cva-fva.pgf")
