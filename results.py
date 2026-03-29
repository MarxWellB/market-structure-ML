import pandas as pd

df = pd.read_csv("historial.csv")

# Filtrar solo oro y plata
df = df[df["Symbol"].isin(["XAUUSD", "XAGUSD"])]

resumen = df.groupby("Symbol")["Profit"].agg([
    "count",
    "sum",
    "mean",
    "max",
    "min"
])

print(resumen)
