from enum import Enum
import requests
import json
from typing import Optional
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

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
		# try convert columns to floats if possible
		for col in df.columns:
			try:
				df[col] = df[col].astype(float)
			except:
				pass
	except:
		raise ValueError("Player history not found.")
	
	if lookback is None:
		return df
	else:
		return df.iloc[-lookback:, :].apply(float_map)
	
def player_data(position: Position, sorted: bool = False):
	"""Retrieve player data on a positional basis with the option to sort.

	Args:
		position (Position): _description_
		sorted (bool): _description_

	Returns:
		_type_: _description_
	"""
	# Query
	response = requests.get("https://fantasy.premierleague.com/api/bootstrap-static/")
	data = json.loads(response.text)
	players = data.get("elements")
	# Create df
	df = pd.DataFrame.from_dict(players)
	# Filter by position and sort
	df = df[df["element_type"]==position.value].apply(float_map)
	if sorted:
		df = df.sort_values(by=["total_points"], ascending=False)
		df = df.reset_index(drop=True)
	return df

def featurizer(df: pd.DataFrame):
	"""Creates features from the dataframe.

	The following features are created:

		1) xG - G
			- < 0: Over-returning or clinical finisher.
			- > 0: Under-performing, not so clinical.
			- = 0: Performing as expected.
		2) xA - A
			- < 0: Over-returning or clinical finisher.
			- > 0: Under-performing, not so clinical.
			- = 0: Performing as expected.

	Args:
		df (pd.DataFrame): _description_
	"""
	df["xG-G"] = df["expected_goals"] - df["goals_scored"]
	df["xA-A"] = df["expected_assists"] - df["assists"]
	return df

def num_players(players: pd.DataFrame, top_n: Optional[int]=None):
	"""Sets either the ``top_n`` players by points, or all players.

	Args:
		top_n (Optional[int], optional): The number of players by points to look at. 
		 Defaults to None.
	"""
	# Calculate number of players to search for
	if top_n is None:
		n = len(players)
	else:
		n = min(top_n, len(players))
	return n

def avg_player_corr(players: pd.DataFrame, regressors: list, top_n: Optional[int]=None):
	"""Calculate the average correlation between exogenous and endogenous
	for positional players.

	Args:
		players (pd.DataFrame): _description_
		regressors (list): _description_
		top_n (int): Will only use this number of players ranked from top by total points.
		 Defaults to None.

	Returns:
		_type_: _description_
	"""
	n = num_players(players, top_n)

	# Sort players by total points
	players.sort_values(by=["total_points"], ascending=False, inplace=True)

	# Calculate correlations
	corr = 0.
	for idx, p in players.iloc[:n,:].iterrows():
		# Query stats
		df = recent_stats(p["id"])
		# Use featurizer
		df = featurizer(df)
		# Calculate uniformly weighted correlation
		corr += (1./n)*df[regressors].select_dtypes('number').corr()
	return corr

def regressions(players: pd.DataFrame, exo: list, endo: str, top_n: Optional[int]=None):
	"""Calculate the multivariate linear regression for a player over their season
	using the regressors supplied by the user. A set of ``n`` regressions will be performed.

	Args:
		players (pd.DataFrame): _description_
		exo (list): _description_
		top_n (int): Will only use this number of players ranked from top by total points.
		 Defaults to None.

	Returns:
		_type_: _description_
	"""
	n = num_players(players, top_n)

	# Sort players by total points
	players.sort_values(by=["total_points"], ascending=False, inplace=True)
	results = []
	# Iterate through players, get detailed information, add features and perform regression.
	for idx, p in players.iloc[:n,:].iterrows():
		# Query stats
		df = recent_stats(p["id"])
		# Use featurizer
		df = featurizer(df)
		# Endogenous
		Y = df[endo].values
		X = sm.add_constant(df[exo].values)
		model = sm.OLS(Y, X)
		results += [model.fit()]
		return results

def forward_form(player_id: int):
	"""Calculates the current form of a forward by linear weights associated
	 with the following characteristics:
		- expected_goals - actual goals
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
	pass

if __name__ == "__main__":

	# Get player stats
	fwds = player_data(Position.FWD)

	# calculate the correlation matrix on the numeric columns
	regressors = [
		"expected_goals", 
		"expected_assists", 
		"ict_index",
		"fixture",
		"xG-G",
		"xA-A",
		"total_points"
		]
	# This includes getting recent data and featurizing the data
	# corr = avg_player_corr(fwds, regressors, 15)
	# sns.heatmap(corr, annot=True)

	# Perform regression analysis for each player over their season.
	# This is used to forward forecast individual player points
	# The prediction intervals may be a function of the stochastic nature or the exogenous variables
	# As well as systematic component.
	exog = ["ict_index", "expected_goals", "fixture"]
	endog = ["goals_scored"]
	results = regressions(fwds, exog, endog, 1)
	insample = results[0].get_prediction().summary_frame(alpha=0.5)
	insample.plot(y=["mean", "obs_ci_lower", "obs_ci_upper"])

	# Plot the original points as well
	plt.scatter(insample.index.values, results[0].model.endog)

	# Produce a forecast with prediction intervals based on estimated ict, excess xG and excess xA
	


	# Plot the selected regressors over the gameweeks
	# g = sns.PairGrid(df[regressors])
	# g.map(sns.scatterplot)
	plt.show()
