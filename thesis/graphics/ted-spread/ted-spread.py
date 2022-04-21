import csv
import pathlib
import datetime
import matplotlib.pyplot as plt

PATH = pathlib.Path(__file__).parent.absolute()

dates, spreads = [], [] 

with open(PATH / "TEDRATE.csv", "r") as file:
    reader = csv.reader(file)
    reader.__next__()
    for date, spread in reader:
        date = datetime.datetime.strptime(date, r"%Y-%m-%d")
        dates.append(date)
        spreads.append(spread)
