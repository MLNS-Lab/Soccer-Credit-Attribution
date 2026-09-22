"""
Functions for Data Collection and Cleaning

Author: Phillip Kornberg
Date: 9.22.26
"""

import os
from statsbombpy import sb
import warnings
import json
from tqdm import tqdm
import pandas as pd

def APICall(competitionId: int, seasonId: int, teamName: str) -> list[str]:
    """
    Retrieve StatsBomb Match Data For a Specific Team and Season.

    Parameters
    ----------
    competitionId : int
        StatsBomb Competition ID To Query.

    seasonId : int
        StatsBomb Season ID To Query.

    teamName : str
        Name Of The Team To Collect Match Data For.

    Returns
    -------
    matchList : list
        List Of Match Files Returned From The API.
    """

    # Supress Warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="statsbombpy")

    # Update Team Name
    updatedTeamName = teamName.replace(" ", "_")

    # Declare Save Folder
    saveFolder = os.path.join("../Data", f"{updatedTeamName}_{seasonId}")
    os.makedirs(saveFolder, exist_ok=True)

    # Find Relevant Matches
    matches = sb.matches(competition_id = competitionId, season_id = seasonId)
    teamMatches = matches[(matches.home_team == teamName) | (matches.away_team == teamName)]
    matchList = []

    # Loop Through Matches
    for idx, match in tqdm(teamMatches.iterrows(), total = len(teamMatches), desc = "Processing Games"):
        match_id = match.match_id
        events = sb.events(match_id = match_id, fmt = "dict")

        if isinstance(events, dict):
            events = list(events.values())

        filePath = os.path.join(saveFolder, f"{match_id}.json")
        with open(filePath, "w") as f:
            json.dump(events, f, indent=2)

        matchList.append(filePath)

    # Returning Match List
    return matchList

def collectDataOneGame(fileName: str, targetTeam: str) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """
    Function to Collect Pass and Shot Data For One Team During One Game.

    Parameters
    ----------
    fileName : str
        Name Of File To Collect Data.

    targetTeam : str
        Name Of Team.

    Returns
    -------
    firstHalf : dataFrame
        Data Frame Containing First Half Events.

    secondHalf : dataFrame
        Data Frame Containing Second Half Events.

    playerInfo1 : dict
        First Half Player Information.

    playerInfo2 : dict
        Second Half Player Information.
    """
    
    # Initializing Arrays
    sender1, receiver1, timestamp1, type1, xG1, location1 = [], [], [], [], [], []
    sender2, receiver2, timestamp2, type2, xG2, location2 = [], [], [], [], [], []

    # Initializing Player Information
    playerInfo1 = {}
    playerInfo2 = {}

    with open(fileName, "r") as f:
        data = json.load(f)

    # Helper Dictionaries to Store Positions and Names
    positions1, names1, ids1 = {}, {}, set()
    positions2, names2, ids2 = {}, {}, set()

    # Active Players During Each Half
    active1 = {}
    active2 = {}

    # Starting Lineups
    foundFirstHalfLineup = False
    foundSecondHalfLineup = False

    # Opponent Goalie Information
    oppGoalie = None

    # Collect Starting Lineups
    for event in data:
        teamName = event.get("team", {}).get("name")
        period = event.get("period")

        # Collect Opponent Goalie
        if teamName != targetTeam:
            if period == 1 and "tactics" in event:
                lineup = event["tactics"].get("lineup", [])
                for p in lineup:
                    if p["position"]["name"] == "Goalkeeper":
                        oppGoalie = p["player"]["id"]
            continue

        # First Half Starting Lineup
        if period == 1 and "tactics" in event and not foundFirstHalfLineup:
            lineup = event["tactics"].get("lineup", [])

            if lineup:
                foundFirstHalfLineup = True

                for p in lineup:
                    pid = p["player"]["id"]
                    position = p["position"]["name"]
                    name = p["player"]["name"]

                    positions1[pid] = position
                    names1[pid] = name
                    ids1.add(pid)

        # Second Half Starting Lineup
        if period == 2 and "tactics" in event and not foundSecondHalfLineup:
            lineup = event["tactics"].get("lineup", [])

            if lineup:
                foundSecondHalfLineup = True

                for p in lineup:
                    pid = p["player"]["id"]
                    position = p["position"]["name"]
                    name = p["player"]["name"]

                    positions2[pid] = position
                    names2[pid] = name
                    ids2.add(pid)

    # If No First Half Lineup, Initialize Empty Dictionaries
    if not foundFirstHalfLineup:
        positions1 = {}
        names1 = {}
        ids1 = set()

    # If No Second Half Lineup, Copy First Half Starting Lineup
    if not foundSecondHalfLineup:
        positions2 = positions1.copy()
        names2 = names1.copy()
        ids2 = ids1.copy()

    # Initialize Active Players
    active1 = {
        pid: {
            "position": positions1[pid],
            "name": names1[pid]
        }
        for pid in ids1
    }

    active2 = {
        pid: {
            "position": positions2[pid],
            "name": names2[pid]
        }
        for pid in ids2
    }

    # Collect Pass and Shot Events
    for i, event in enumerate(data):
        teamName = event.get("team", {}).get("name")
        period = event.get("period")

        # Ignore Events That Are Not From Target Team
        if teamName != targetTeam:
            continue

        pid = event.get("player", {}).get("id")
        time = event.get("minute", 0) * 60 + event.get("second", 0)
        location = event.get("location")

        # First Half
        if period == 1:

            # Collect Substitution Information
            if event["type"]["name"] == "Substitution":

                oldPid = event["player"]["id"]
                sub = event["substitution"]["replacement"]

                newPid = sub["id"]
                newName = sub["name"]

                # Get Position Of Player Coming Off
                position = active1.get(oldPid, {}).get("position")

                # Remove Player Coming Off
                if oldPid in active1:
                    del active1[oldPid]

                # Add Player Coming On
                active1[newPid] = {
                    "position": position,
                    "name": newName
                }

                # Store Substitute In Player Information
                positions1[newPid] = position
                names1[newPid] = newName
                ids1.add(newPid)

                continue

            # Pass
            if "pass" in event and pid in active1:
                receiver = event["pass"].get("recipient", {}).get("id", -2)

                # Only Keep Passes To Players Currently On The Pitch
                if receiver not in active1:
                    continue

                sender1.append(pid)
                receiver1.append(receiver)
                timestamp1.append(time)
                type1.append("Pass")
                xG1.append(0)
                location1.append(location)

            # Shot
            if "shot" in event and pid in active1:
                sender1.append(pid)
                receiver1.append(-1)
                timestamp1.append(time)
                type1.append("Shot")

                xG = round(event["shot"]["statsbomb_xg"], 2)

                xG1.append(xG)
                location1.append(location)

        # Second Half
        if period == 2:

            # Collect Substitution Information
            if event["type"]["name"] == "Substitution":

                oldPid = event["player"]["id"]
                sub = event["substitution"]["replacement"]

                newPid = sub["id"]
                newName = sub["name"]

                # Get Position Of Player Coming Off
                position = active2.get(oldPid, {}).get("position")

                # Remove Player Coming Off
                if oldPid in active2:
                    del active2[oldPid]

                # Add Player Coming On
                active2[newPid] = {
                    "position": position,
                    "name": newName
                }

                # Store Substitute In Player Information
                positions2[newPid] = position
                names2[newPid] = newName
                ids2.add(newPid)

                continue

            # Pass
            if "pass" in event and pid in active2:
                receiver = event["pass"].get("recipient", {}).get("id", -2)

                # Only Keep Passes To Players Currently On The Pitch
                if receiver not in active2:
                    continue

                sender2.append(pid)
                receiver2.append(receiver)
                timestamp2.append(time)
                type2.append("Pass")
                xG2.append(0)
                location2.append(location)

            # Shot
            if "shot" in event and pid in active2:
                sender2.append(pid)
                receiver2.append(-1)
                timestamp2.append(time)
                type2.append("Shot")

                xG = round(event["shot"]["statsbomb_xg"], 2)

                xG2.append(xG)
                location2.append(location)

    # Add Goal Node
    positions1[-1] = "Goal"
    names1[-1] = "Goal Node"
    ids1.add(-1)

    positions2[-1] = "Goal"
    names2[-1] = "Goal Node"
    ids2.add(-1)

    # Build Player Information Dictionary
    playerInfo1[targetTeam] = {
        pid: {
            "position": positions1[pid],
            "name": names1[pid]
        }
        for pid in ids1
    }

    playerInfo2[targetTeam] = {
        pid: {
            "position": positions2[pid],
            "name": names2[pid]
        }
        for pid in ids2
    }

    # Format Dataframes
    firstHalf = pd.DataFrame({
        "Sender": sender1,
        "Receiver": receiver1,
        "Timestamp": timestamp1,
        "Type": type1,
        "xG": xG1,
        "Location": location1
    })

    secondHalf = pd.DataFrame({
        "Sender": sender2,
        "Receiver": receiver2,
        "Timestamp": timestamp2,
        "Type": type2,
        "xG": xG2,
        "Location": location2
    })

    return firstHalf, secondHalf, playerInfo1, playerInfo2