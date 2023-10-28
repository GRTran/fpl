import get_stats
import pandas as pd
import numpy as np

df = pd.read_csv("summary_stats.csv")
print(df[df[("Unnamed: 2_level_0", "Pos")] == "DF"])
