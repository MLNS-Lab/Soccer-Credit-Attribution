"""
Functios For Data Generation Across Entire Season

Author: Phillip Kornberg
Date: 9.22.26
"""

from src.modelFitting import shotResponsibility
from src.analysis import updatedPlayerInfo, playerInfoDataFrame
import pandas as pd
import numpy as np

def fullSeasonStatistics(matches: list, gameDfs: list, teamName: str, playerInformation: dict,
    mu: np.ndarray, beta: int, alpha: np.ndarray, allGamePairs: dict) -> tuple[pd.DataFrame, list[dict]]:
    """
    Function To Generate Shot Responsibility Data For Multiple Matches.

    Parameters
    ----------
    matches : list
        Array Of Match File Paths.

    gameDfs : list
        Array Of Game Data Frames Returned From MultiGameCollection().

    teamName : str
        Name Of Team.

    playerInformation : dict
        Global Player Information Dictionary.

    mu : np.ndarray
        Matrix Of Baseline Rates (Mu).

    beta : int
        Decay Value (Beta) Not Scaled.

    alpha : np.ndarray
        Matrix Of Alpha Values.

    allGamePairs : dict
        Returned From ProcessData().

    Returns
    -------
    df : pandas.DataFrame
        Data Frame With Stats From Multiple Matches.

    allResponsibilities : list[dict]
        All Shot Responsibilities Across Matches.
    """

    # Initializing Variables
    matchNumber = 0
    fullData = []
    allResponsibilities = []

    # Looping Through Matches
    for match in matches:
        responsibilities_list, shotLists = shotResponsibility(
            [gameDfs[matchNumber]], 
            mu,
            beta,
            alpha,
            allGamePairs,
            teamName,
            playerInformation
        )

        # unpack single-game outputs
        responsibilities = responsibilities_list[0]
        shotList = shotLists[0]

        updatedInfo = updatedPlayerInfo(gameDfs[matchNumber], responsibilities, playerInformation)
        dfStats = playerInfoDataFrame(updatedInfo)

        fullData.append(dfStats)
        allResponsibilities.append(responsibilities)

        matchNumber += 1


    # Merging Data Frames
    mergedDataFrame = pd.concat(fullData, ignore_index = True)

    # Updating Grouping
    df = mergedDataFrame.groupby("Name", as_index = False).agg({
            "Name": "first",
            "Position": "first",
            "xGI": "sum",  
            "Games Played": "sum", 
            "Team Name": "first"
        })

    # Rounding and Returning Data Frame
    df = df.round(3)
    df["Average xGI"] = round(df["xGI"] / df["Games Played"], 3)

    return df, allResponsibilities