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
        quarter = date[:2]
        year = date[-2:]
        date = f"{quarter} '{year}"

        dates.append(date)
        spreads.append(float(spread.replace(",", "")))


fig, ax = plt.subplots(figsize=(7, 3))

@ticker.FuncFormatter
def million_formatter(x, pos):
    sign = "-" if x < 0 else ""
    x_in_millions = abs(x * 1e-6)
    return f"{sign}\${x_in_millions:.0f}M"

ax.set_ylim([-1e9, 0.4e9])
ax.yaxis.set_major_formatter(million_formatter)

ax.tick_params(axis="x", rotation=45)
ax.set_title("JP Morgan Credit Adjustments \& Other")

ax.plot(dates, spreads, color=[i / 255 for i in [255, 68, 61]])

plt.tight_layout()
plt.savefig(PATH / "jp-morgan-cva-fva.pgf")
