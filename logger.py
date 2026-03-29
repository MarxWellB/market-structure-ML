import csv
from datetime import datetime

def log_trade(data):
    with open("live_trades.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow(), *data
        ])
