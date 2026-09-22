"""
Functions for Model Preparation and Data Processing

Author: Phillip Kornrg
Date: 9.22.26
"""

import sys
import numpy as np
import copy
import pandas as pd
from collections import defaultdict
from collections import Counter
from src.dataCollection import collectDataOneGame

def dataInformation(dataFrame: pd.DataFrame, playerInformation: dict) -> tuple[int, int, int, dict]:
    """
    Function To Display Information About DataFrame.

    Parameters
    ----------
    dataFrame : pandas.DataFrame
        Cleaned DataFrame Containing (Sender, Receiver, Timestamp).

    playerInformation : dict
        Information About Players.

    Returns
    -------
    numNodes : int
        Number Of Nodes In DataFrame.

    numEvents : int
        Number Of Events In DataFrame.

    duration : int
        Duration Of DataFrame.

    eventDict : dict
        Dictionary Of All Events.
    """

    # Creating Event Dictionary
    eventDict = defaultdict(list)

    # Adding Unique Indexes to Set and Information to Event Dictionary
    for _, row in dataFrame.iterrows():
        key = (int(row["Sender"]), int(row["Receiver"]))
        eventDict[key].append(int(row["Timestamp"]))

    # Calculating Number of Nodes
    numNodes = 0
    for team_dict in playerInformation.values():
        for player in team_dict.values():
            numNodes += 1

    # Finding Number of Events in Event Dictionary
    numEvents = 0
    for _, t in eventDict.items():
        numEvents = numEvents + len(t)
    
    # Calculating Duration of DataFrame
    duration = int(dataFrame["Timestamp"][len(dataFrame) - 1])

    return numNodes, numEvents, duration, eventDict

def createAggAdj(numNodes: int, dataFrame: pd.DataFrame) -> np.ndarray:
    """
    Function To Generate Aggregated Adjacency Matrix From DataFrame.

    Parameters
    ----------
    numNodes : int
        Number Of Nodes In DataFrame.

    dataFrame : pandas.DataFrame
        Cleaned DataFrame Containing (Sender, Receiver, Timestamp).

    Returns
    -------
    aggAdj : np.ndarray
        Aggregated Adjacency Matrix.
    """

    # Initialzing Event Dictionary and Adjacency Matrix
    eventDict = defaultdict(list)
    aggAdj = np.zeros((numNodes, numNodes))

    # Adding Information to Dictionary
    for _, row in dataFrame.iterrows():
        key = (int(row["Sender"]), int(row["Receiver"]))
        eventDict[key].append(int(row["Timestamp"]))
    
    # Creating Aggregated Adjacency Matrix
    for (x, y), t in eventDict.items():
        aggAdj[x, y] = len(t)

    return aggAdj

def buildGlobalPlayerDict(matches: list, teamName: str) -> dict:
    """
    Function To Build Global Player Dictionary For Consistent Player Mapping.

    Parameters
    ----------
    matches : list
        List Of File Paths To All Matches.

    teamName : str
        Name Of Team.

    Returns
    -------
    globalPlayers : dict
        Dictionary Of Player Mappings.
    """

    # Initializing Global Players Dictionary 
    globalPlayers = {teamName: {}}

    # Looping Through Matches
    for match in matches:
        _, _, playerInfo1, playerInfo2 = collectDataOneGame(match, teamName)

        teamPlayers = {}
        teamPlayers.update(playerInfo1.get(teamName, {}))
        teamPlayers.update(playerInfo2.get(teamName, {}))

        globalPlayers[teamName].update(teamPlayers)

    # Assigning Mapping
    players = globalPlayers[teamName]
    sortedIDs = sorted(pid for pid in players if pid != -1)

    for idx, pid in enumerate(sortedIDs):
        players[pid]["index"] = idx

    # Giving Goal Node New Index
    goalIndex = len(sortedIDs)
    players[-1]["index"] = goalIndex

    return globalPlayers

def multiGameCollection(matches: list, teamName: str, globalPlayerDict: dict | None = None) -> tuple[list[pd.DataFrame], dict]:
    """
    Function To Collect Data Across Multiple Matches.

    Parameters
    ----------
    matches : list
        List Of File Paths To All Matches.

    teamName : str
        Name Of Team.

    globalPlayerDict : dict | None
        Optional Precomputed Global Player Dictionary.

    Returns
    -------
    gameDfs : list[pandas.DataFrame]
        Array Of DataFrames (One For Each Game).

    globalPlayerDict : dict
        Dictionary Of Mapping.
    """

    # Initializing Array
    gameDfs = []

    # Building Global Player Dictionary
    if globalPlayerDict is None:
        globalPlayerDict = buildGlobalPlayerDict(matches, teamName)

    # Build Player Mapping (ID to Index)
    newDict = {pid: info["index"] for pid, info in globalPlayerDict[teamName].items()}

    # Processing Each Match
    for match in matches:
        firstHalf, secondHalf, _, _ = collectDataOneGame(match, teamName)
        
        # Offseting Second Half Time
        secondHalf["Timestamp"] = secondHalf["Timestamp"] + 600
        matchDf = pd.concat([firstHalf, secondHalf], ignore_index = True)

        # Mapping Player ID to Indexes Consistently
        matchDf["Sender"] = matchDf["Sender"].map(newDict)
        matchDf["Receiver"] = matchDf["Receiver"].map(newDict)

        # Dropping Rows that Can't Be Mapped
        matchDf = matchDf.dropna(subset=["Sender", "Receiver"])

        # Convert indices to int
        matchDf["Sender"] = matchDf["Sender"].astype(int)
        matchDf["Receiver"] = matchDf["Receiver"].astype(int)

        gameDfs.append(matchDf)

    return gameDfs, globalPlayerDict

def passCounts(dataFrames: list[pd.DataFrame]) -> Counter:
    """
    Function To Count Total Passes For Each (Sender, Receiver) Pair Across All Games.

    Parameters
    ----------
    dataFrames : list[pandas.DataFrame]
        List Of Match DataFrames.

    Returns
    -------
    passCount : Counter
        Counts Of Passes Per Pair.
    """

    # Initializing Empty Array
    allPasses = []
    passCount = Counter()

    # Looping Thruogh Matches
    for df in dataFrames:
        if "Type" in df.columns:
            df = df[df["Type"] == "Pass"]

        # Counting Occurances of Pair
        pairCounts = df.groupby(["Sender", "Receiver"]).size()

        for (sender, receiver), count in pairCounts.items():
            passCount[(sender, receiver)] += int(count)

    return passCount

def processData(dataFrames: list[pd.DataFrame], combinationCount: int, playerInformation: dict, teamName: str) -> tuple[list[list[np.ndarray]], float, list[dict]]:
    """
    Function To Transform Data Into Hawkes-Ready Format.

    Parameters
    ----------
    dataFrames : list[pandas.DataFrame]
        Array Of DataFrames From MultiGameCollection().

    combinationCount : int
        Number Of Frequently Occurring Pairs To Keep.

    playerInformation : dict
        Global Dictionary Of Player Information.

    teamName : str
        Name Of Team Being Processed.

    Returns
    -------
    gameList : list
        Hawkes-Ready Event Sequences.

    T : float
        Maximum Timestamp For Model Fitting.

    cleanedGamePairs : list[dict]
        Clean Aligned Dictionaries.
    """

    # Copying Data
    copyData = [df.copy() for df in dataFrames] 

    # Finding Max Timestamp
    T = float(max(df["Timestamp"].max() for df in copyData))

    # Transform To Dictionaries
    allGamePairs = []

    for gameDf in copyData:
        gameDictionary = (
            gameDf
            .groupby(['Sender', 'Receiver'])['Timestamp']
            .apply(lambda x: np.sort(x.to_numpy(dtype = np.float64)))
            .to_dict()
        )
        allGamePairs.append(gameDictionary)

    # Count Pair Frequency
    passCount = passCounts(dataFrames)
    frequentKeys = set([k for k, v in passCount.most_common(combinationCount)])

    # Building All kepys
    allKeys = set()
    for gameDictionary in allGamePairs:
        allKeys.update(gameDictionary.keys())

    # Keeping Only Important Keys
    canonicalKeys = sorted(list(
        key for key in allKeys
        if key in frequentKeys or key[1] == playerInformation[teamName][-1]["index"]
    ))

    gameList = []
    cleanedGamePairs = []

    for gameDictionary in allGamePairs:

        matchList = []
        matchDict = {}

        for key in canonicalKeys:
            if key in gameDictionary:
                arr = gameDictionary[key]
            else:
                arr = np.array([], dtype=np.float64)

            matchList.append(arr)
            matchDict[key] = arr

        gameList.append(matchList)
        cleanedGamePairs.append(matchDict)

    return gameList, T, cleanedGamePairs