"""
Functions for Analysis

Author: Phillip Kornberg
Date: 9.22.26
"""

import pandas as pd
from sklearn.metrics import ndcg_score
import numpy as np

def interpretShots(shotResponsibilityDict: dict, dataFrame, playerInformation: dict) -> None:
    """
    Function To Help Visualize Shot Responsibility Dictionary For Single Game.

    Parameters
    ----------
    shotResponsibilityDict : dict
        Dictionary Of Shot Responsibilities.

    dataFrame : dataFrame
        Data Frame From First Or Second Half.

    playerInformation : dict
        Player Information From Same Half As DataFrame.

    Returns
    -------
    None
        Prints Out Shot Responsibility Dictionary.
    """

    # Looping Through Responsibility Dictionary
    for shot, responsibilities in shotResponsibilityDict.items():
        time = dataFrame.iloc[shot]["Timestamp"]
        shotMin = time // 60
        shotSeconds = time % 60
        shooterIndex = dataFrame.iloc[shot]["Sender"]
        xG = dataFrame.iloc[shot]["xG"]

        # Extracting Shooter Name, Team and Position
        for team, players in playerInformation.items():
            for _, info in players.items():
                if info.get('index') == shooterIndex:
                    shooterTeamName = team
                    shooterName = info.get('name')
                    shooterPosition = info.get('position')
        
        # Printing Out Shot Information
        print(f"[{shot}: xG = {xG}] Shot at {shotMin}:{shotSeconds} by {shooterName} ({shooterPosition} {shooterTeamName}):")

        # Looping Through All Responsibilities
        for event, responsibilityLevel in responsibilities.items():
            eventTime = dataFrame.iloc[event]["Timestamp"]
            eventMin = eventTime // 60
            eventSeconds = eventTime % 60
            eventIndex = dataFrame.iloc[event]["Sender"]
            eventType = dataFrame.iloc[event]["Type"]
            passReceiver = dataFrame.iloc[event]["Receiver"]

            # Finding Information on Sender
            for team, players in playerInformation.items():
                for player_id, info in players.items():
                    if info.get('index') == eventIndex:
                        eventTeamName = team
                        eventName = info.get('name')
                        eventPosition = info.get('position')
            
            # Finding Information on Receiver
            for team, players in playerInformation.items():
                for player_id, info in players.items():
                    if info.get('index') == passReceiver:
                        eventTeamName2 = team
                        eventName2 = info.get('name')
                        eventPosition2 = info.get('position')
            
            # Priniting Out Pass Responsibility Information
            if event == -1:
                print(f"[{event}: {responsibilityLevel}] Spontaneous Shot Responsibility Level")
            elif eventType == "Pass":
                print(f"[{event}: {responsibilityLevel}] Pass at {eventMin}:{eventSeconds} by {eventName} ({eventPosition}, {eventTeamName}) to {eventName2} ({eventPosition2}, {eventTeamName2})")
            else:
                print(f"[{event}: {responsibilityLevel}] Shot at {eventMin}:{eventSeconds} by {eventName} ({eventPosition}, {eventTeamName})")

        print("")
    


def updatedPlayerInfo(dataFrame, shotResponsibility: dict, playerInformation: dict) -> dict:
    """
    Function To Update Player Info With Shooting Chances Created, Expected Goal Involvements, Passes, Shots, And Interceptions For Each Player.

    Parameters
    ----------
    dataFrame : dataFrame
        Data Frame Returned From CollectData().

    shotResponsibility : dict
        Dictionary Of Shot Responsibilities (Top 5 Values).

    playerInformation : dict
        Dictionary Of Player Information.

    Returns
    -------
    playerInformation : dict
        Updated Player Information With Metrics Fields Added.

    Metrics Calculated
    ------------------
    - Expected Goal Involvements (xGI)
    """

    # Updating Dictionary With Fields
    for team, players in playerInformation.items():
        for playerID, playerInfo in players.items():
            playerInfo["expectedGoalInvolvement"] = 0
            playerInfo["gamesPlayed"] = 0

    # Creating Index Map
    indexMap = {}
    for team, players in playerInformation.items():
        for playerId, playerInfo in players.items():
            index = playerInfo.get("index")
            indexMap[index] = (team, playerId)

    # Track Players in Game    
    playersInGame = set(dataFrame["Sender"].unique()) | set(dataFrame.get("Receiver", pd.Series([])).dropna().unique())

    # Update Games Played
    for index in playersInGame:
        if index in indexMap:
            team, playerId = indexMap[index]
            if "gamesPlayed" not in playerInformation[team][playerId]:
                playerInformation[team][playerId]["gamesPlayed"] = 1
            else:
                playerInformation[team][playerId]["gamesPlayed"] += 1

    # Updating Chances Created and Expected Goals By Looping Through Shot Responsibility Dictionary
    for key, info in shotResponsibility.items():
        xG = dataFrame.loc[key, "xG"]
        shooter = int(dataFrame.loc[key, "Sender"])
        for eventIndex, probability in info.items():
            if eventIndex >= 0:
                sender = int(dataFrame.loc[eventIndex, "Sender"])
                if sender in indexMap:
                    team, playerId = indexMap[sender]
                    team2, _ = indexMap[shooter]
                    
                    # Checking That Shooter and Passer Were on Same Team
                    if team == team2:
                        playerInformation[team][playerId]["expectedGoalInvolvement"] += round(probability * xG, 3)

    # Returning Updated Dictionary
    return playerInformation

def playerInfoDataFrame(playerInformation: dict) -> pd.DataFrame:
    """
    Function To Convert Player Information Dictionary To DataFrame.

    Parameters
    ----------
    playerInformation : dict
        Dictionary Of Player Information Returned By UpdatedPlayerInfo().

    Returns
    -------
    information : dataFrame
        DataFrame Containing All Information.
    """

    # Adding Information to Rows
    rows = []
    for team, players in playerInformation.items():
        for playerID, stats in players.items():

            # Determining Stats
            row = {
                "Index": stats.get("index"),
                "Name": stats.get("name"),
                "Position": stats.get("position"),
                "xGI": stats.get("expectedGoalInvolvement"),
                "Games Played": stats.get("gamesPlayed"),
                "Team Name": team
            }
            rows.append(row)


    # Creating Data Frame
    information = pd.DataFrame(rows)
    information = information[information["Index"] >= 0]
    information = information.round(3)
    information = information.sort_values(by = "xGI", ascending = False)
    information = information[information["Name"] != "Goal Node"]

    return information

def predictiveMetrics(pairs: list, gameData: list[pd.DataFrame], alpha: np.ndarray) -> None:
    """
    Function To Evaluate Model's Pass Predictions

    Parameters
    ----------
    pairs : list
        List of Frequent Pairs

    gameData : list[pd.DataFrame]
        List of Game Data  

    alpha : np.ndarray
        Matrix of Alpha Values for Pairs

    Returns
    -------
    None
        Prints out Normalized Discounted Cumulative Gain (NDCG), Top 1 and Top 5 Accuracy
    """ 

    # Creating Pair Index Lookup and Initializing Arrays
    pairToIdx = {pair: idx for idx, pair in enumerate(pairs)}
    allNdcgs = []
    top1Correct = []
    top5Correct = []

    # Looking at the Next 3 Passes
    futureWindow = 3
    relevanceScores = [3, 2, 1]

    # Looping Through Games
    for df in gameData:

        # Finding Valid Game Pairs
        gamePairs = list(zip(df['Sender'], df['Receiver']))
        validPairs = set(gamePairs)
        validIndices = [pairToIdx[p] for p in validPairs if p in pairToIdx]

        # Finding Next Passes
        for i in range(len(gamePairs) - futureWindow):

            # Making Sure Pairs are Frequent, Not Shots, and Not Gaps in Play 
            currentPair = gamePairs[i]
            futurePair = gamePairs[i + 1]

            if currentPair not in pairToIdx:
                continue
            if currentPair[1] == 25:
                continue
            if futurePair not in pairToIdx:
                continue
            if currentPair[1] != futurePair[0]:
                continue

            currentIdx = pairToIdx[currentPair]

            # Extracting Alpha Values and Updating Scores
            globalScores = alpha[:, currentIdx].copy()
            yScore = np.full(len(pairs), -1e-9)
            yScore[validIndices] = globalScores[validIndices]
            yTrue = np.zeros(len(pairs))

            # Finding Actual Future Passes
            actualFutureIndices = []

            # Finding Actual Next Pass
            for step in range(futureWindow):
                futurePair = gamePairs[i + step + 1]

                # Making Sure Future Pair is Frequent and Valid
                if futurePair in pairToIdx:
                    futureIdx = pairToIdx[futurePair]
                    actualFutureIndices.append(futureIdx)
                    yTrue[futureIdx] = relevanceScores[step]

            # Updating  Normalized Discounted Cumulative Gain 
            ndcg = ndcg_score([yTrue], [yScore], k = 5)
            allNdcgs.append(ndcg)

            # Calculating Other Metrics
            predictedIdx = np.argmax(yScore)
            top1Correct.append(predictedIdx in actualFutureIndices)
            top5Indices = np.argsort(yScore)[-5:]
            top5Correct.append(any(idx in top5Indices for idx in actualFutureIndices))

    meanNdcg = np.mean(allNdcgs)
    top1Accuracy = np.mean(top1Correct)
    top5Accuracy = np.mean(top5Correct)

    # Printing Out Metrics
    print("Mean NDCG@5:", round(meanNdcg, 3))
    print("Top 1 Accuracy:", round(top1Accuracy, 3))
    print("Top 5 Accuracy:", round(top5Accuracy, 3))
    print(f"Total Events: {len(allNdcgs)}")

def passPrediction(pairs: list, gameData: list[pd.DataFrame], alpha: np.ndarray, passIdx: int) -> None:
    """
    Function To Print Out Predicted Future Pass and Actual Future Pass

    Parameters
    ----------
    pairs : list
        List of Frequent Pairs

    gameData : list[pd.DataFrame]
        List of Game Data  

    alpha : np.ndarray
        Matrix of Alpha Values for Pairs

    passIdx : int
        Index of Current Pass 
    

    Returns
    -------
    None
        Prints out Current Pass, Top 5 Pass Predictions, Actual Future Pass, and NDCG@5
    """ 

    # Selecting One Game for Evaluation and Future Window
    df = gameData[0]
    futureWindow = 3
    scores = [3, 2, 1]

    # Creating Pair Index Lookup 
    pairIndex = {pair: idx for idx, pair in enumerate(pairs)}
    idxPairs = {idx: pair for idx, pair in enumerate(pairs)}
    gamePairs = list(zip(df['Sender'], df['Receiver']))

    # Selecting and Printing Current Pair
    currentPair = gamePairs[passIdx]
    print(f"Current Pass: {currentPair}")

    # Making Sure Current Pair is Valid
    if currentPair not in pairIndex:
        print("Invalid Pair")
    elif currentPair[1] == 25:
        print("Invalid Pair")
    elif passIdx + futureWindow >= len(gamePairs):
        print("Not Enough Future Passes")
    else:

        # Selecting Future Pair    
        futurePair = gamePairs[passIdx + 1]

        # Making Sure Future Pair is Valid
        if futurePair not in pairIndex:
            print("Invalid Future Pair")
        elif currentPair[1] != futurePair[0]:
            print("Invalid Future Pair")
        else:

            # Finding Top 5 Passes
            currentIdx = pairIndex[currentPair]
            validPairs = set(gamePairs)
            validIndices = [pairIndex[p] for p in validPairs if p in pairIndex]

            # Finding Alpha Scores
            globalScores = alpha[:, currentIdx].copy()

            # Removing Pairs Not In Current Game
            yScore = np.full(len(pairs), -1e-9)
            yScore[validIndices] = globalScores[validIndices]
            yTrue = np.zeros(len(pairs))

            # Finding Future Pass
            actualFuture = []
            for step in range(futureWindow):
                futurePair = gamePairs[passIdx + step + 1]
                actualFuture.append(futurePair)
                if futurePair in pairIndex:
                    futureIdx = pairIndex[futurePair]
                    yTrue[futureIdx] = scores[step]

            # Calculating NDCG
            ndcg = ndcg_score([yTrue], [yScore], k = 5)

            # Printing Out Top Passes
            topIndexes = np.argsort(yScore)[::-1][:5]
            print("\nTop Predicted Next Passes:")
            for rank, idx in enumerate(topIndexes, start=1):
                pair = idxPairs[idx]
                score = yScore[idx]
                print(f"{rank}. {pair}: {score:.3f}")

            # Printing Out Actual Next Pass and NDCG
            print("\nActual Next Pass:")
            for i, pair in enumerate(actualFuture, start = 1):
                print(f"{i}. {pair}")
            print(f"\nNDCG@5: {ndcg:.3f}")
