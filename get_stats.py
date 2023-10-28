import pandas as pd
import numpy as np

urls={
	"arsenal": "https://fbref.com/en/squads/18bb7c10/Arsenal-Stats",
	"aston_villa": "https://fbref.com/en/squads/8602292d/Aston-Villa-Stats",
	"bournemouth": "https://fbref.com/en/squads/4ba7cbea/Bournemouth-Stats",
	"brentford": "https://fbref.com/en/squads/cd051869/Brentford-Stats",
	"brighton": "https://fbref.com/en/squads/d07537b9/Brighton-and-Hove-Albion-Stats",
	"burnley": "https://fbref.com/en/squads/943e8050/Burnley-Stats",
	"chelsea": "https://fbref.com/en/squads/cff3d9bb/Chelsea-Stats",
	"palace": "https://fbref.com/en/squads/47c64c55/Crystal-Palace-Stats",
	"everton": "https://fbref.com/en/squads/d3fd31cc/Everton-Stats",
	"fulham": "https://fbref.com/en/squads/fd962109/Fulham-Stats",
	"liverpool": "https://fbref.com/en/squads/822bd0ba/Liverpool-Stats",
	"luton": "https://fbref.com/en/squads/e297cd13/Luton-Town-Stats",
	"man_city": "https://fbref.com/en/squads/b8fd03ef/Manchester-City-Stats",
	"man_united": "https://fbref.com/en/squads/19538871/Manchester-United-Stats",
	"newcastle": "https://fbref.com/en/squads/b2b47a98/Newcastle-United-Stats",
	"forest": "https://fbref.com/en/squads/e4a775cb/Nottingham-Forest-Stats",
	"sheffield": "https://fbref.com/en/squads/1df6b87e/Sheffield-United-Stats",
	"spurs": "https://fbref.com/en/squads/361ca564/Tottenham-Hotspur-Stats",
	"west_ham": "https://fbref.com/en/squads/7c21e445/West-Ham-United-Stats",
	"wolves": "https://fbref.com/en/squads/8cec06e1/Wolverhampton-Wanderers-Stats"
	}

def pull_df(url):
	return pd.read_html(url, attrs={"id": "stats_standard_9"})[0]


def pull_standard_data():
	res = {}
	for key, url in urls.items():
		res[key] = pull_df(url)
	return res


def save_to_csv(info: list) -> None:
	aggregated = pd.DataFrame()
	for key, df in info.items():
		df.loc[:, "Team"] = key
		df = df[np.logical_and(df[("Unnamed: 0_level_0", "Player")] != "Squad Total", df[("Unnamed: 0_level_0", "Player")] != "Opponent Total")]
		aggregated = pd.concat((aggregated, df), ignore_index=True)
	aggregated.to_csv("summary_stats.csv")
	



if __name__ == "__main__":
	info = pull_standard_data()
	save_to_csv(info)
