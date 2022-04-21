import csv
import pathlib
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import PercentFormatter

plt.style.use("ggplot")
plt.ion()

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

PATH = pathlib.Path(__file__).parent.absolute()

dates, spreads = [], [] 

with open(PATH / "TEDRATE.csv", "r") as file:
    reader = csv.reader(file)
    # Pop the header
    reader.__next__()
    for date, spread in reader:
        line_is_empty = (spread == ".")
        if not line_is_empty:
            date = datetime.datetime.strptime(date, r"%Y-%m-%d")
            dates.append(date)
            spreads.append(float(spread))


fig, ax = plt.subplots(figsize=(7, 4))

ax.set_ylim([0, 5])
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=1, is_latex=True))
ax.set_xlim([datetime.date(2004, 1, 1), max(dates)])

ax.set_title("TED Spread from 2004 to " + str(max(dates).year))
ax.set_ylabel("Spread")
ax.set_xlabel("Date")

ax.plot(dates, spreads, color=[i / 255 for i in [255, 68, 61]])
plt.savefig(PATH / "ted-spread.pgf")
