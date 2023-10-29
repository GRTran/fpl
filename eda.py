from enum import Enum
import requests
import json
from typing import Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Position(Enum):
	GKP = 1
	DEF = 2
	MID = 3
	FWD = 4

def float_map(x):
	try:
		return float(x)
	except:
		return x

def recent_stats(player_id: int, lookback: Optional[int] = None):
	"""Based on a player ID and lookback period, obtain recent player statistics.

	Args:
		player_id (int): The identifier of the player.
		lookback (int): The number of games in the past to look at.
	"""

	html = requests.get(f"https://fantasy.premierleague.com/api/element-summary/{player_id}/")
	data = json.loads(html.text)

	season_data = data.get("history")
	try:
		df = pd.DataFrame.from_dict(season_data)
	except:
		raise ValueError("Player history not found.")
	
	if lookback is None:
		return df
	else:
		return df.iloc[-lookback:, :].apply(float_map)
	
def player_data(position: Position):
	response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
	data = json.loads(response.text)
	players = data.get("elements")
	df = pd.DataFrame.from_dict(players)
	return df[df["element_type"]==position.value].apply(float_map)
	

def forward_form(player_id: int):
	"""Calculates the current form of a forward by linear weights associated
	 with the following characteristics:
		- expected_goals
		- expected_assists
		- goals_scored
		- assists
	
	The true response is the total points.

	We try and regress these factors to the total points to see their importance.
	
	Then by predicting future parameters using another model, we can forecast the 
	points return of each player.

	The model can be backtested using the previous season if data is available.

	Args:
		player_id (int): _description_
	"""

if __name__ == "__main__":

	# Get player stats
	fwds = player_data(Position.FWD)

	# Tester for Julian Alvarez
	p = fwds[fwds["web_name"]=="J.Alvarez"]["id"].reset_index(drop=True)
	df = recent_stats(p[0])

	# calculate the correlation matrix on the numeric columns
	regressors = ["expected_goals", "expected_assists", "ict_index", "total_points"]
	corr = df[regressors].select_dtypes('number').corr()
	sns.heatmap(corr)
	plt.show()
