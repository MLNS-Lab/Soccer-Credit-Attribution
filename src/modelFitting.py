"""
Functions for Hawkes Torche Parameter Generation and Shot Responsibilities

Author: Phillip Kornberg
Date: 9.22.26
"""

import torch
from tqdm import tqdm
from hawkes.utils.utils import EventSequence
from hawkes import models
from hawkes.models.penalty import NuclearPenalty, L2Penalty
import numpy as np
import pandas as pd
import random

def fitHawkesTorch(processedData: list, pairData: dict, beta: float, maxIterations: int, device: str | None = None
                   ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, list]:
    """
    Train A Multivariate Hawkes Process Model On Game Sequences.

    Parameters
    ----------
    processedData : list
        List Of Game Event Sequences. Each Element Contains
        Timestamps For Each Event Type (Pair).

    pairData : dict
        Dictionary Containing Pair Information. pairData[0].keys()
        Gives The List Of Event Types.

    beta : float
        Decay Rate (Gamma) For The Hawkes Process Excitation Kernel.

    maxIterations : int
        Maximum Number Of Training Iterations (Steps).

    device : str | None
        Device To Run On ('cuda' Or 'cpu'). Auto-Detects If None.

    Returns
    -------
    mu : np.ndarray
        Baseline Intensity Values For Each Event Type.

    alpha : np.ndarray
        Excitation Matrix (Influence Of Event i On Event j).

    gamma : torch.Tensor
        Decay Rate Parameter.

    sequences : list
        List Of EventSequence Objects Created From The Data.
    """
    
    # Setting Seed
    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Auto-Detect Device If Not Specified
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Build Game Sequences
    pairs = list(pairData[0].keys())
    M = len(pairs)
    print(f"Total Event Types (M): {M}")
    
    sequences = []
    for gameIndex in tqdm(range(len(processedData)), desc = "Building Independent Sequences"):
        gameTimes = processedData[gameIndex]
        
        # Ensure Game Has The Correct Number Of Event Streams
        if len(gameTimes) != M:
            continue
        
        # Create All Time and Marks
        allTimes = []
        allMarks = []
        
        for pairIndex, timestamps in enumerate(gameTimes):
            if timestamps is None or len(timestamps) == 0:
                continue
            allTimes.extend(timestamps)
            allMarks.extend([pairIndex] * len(timestamps))
        
        if len(allTimes) == 0:
            continue
        
        # Create Tensors Directly On The Target Device
        ti = torch.tensor(allTimes, dtype = torch.float32, device = device)
        mi = torch.tensor(allMarks, dtype = torch.long, device = device)
        
        # Sort By Time 
        sortIdx = torch.argsort(ti)
        ti = ti[sortIdx]
        mi = mi[sortIdx]
        
        # Create The EventSequence Object For This Specific Game
        seq = EventSequence(ti = ti, mi = mi, T = float(ti.max().item()), M = M)
        sequences.append(seq)
    
    # Initialize Model
    hawkes_model = models.Hawkes(
        gamma = torch.tensor([beta], dtype = torch.float32),
        gamma_param = False, 
        base_process = models.Poisson(M = M),
    ).to(device)
    
    # Train Hawkes Model
    lr = 0.05
    optimizer = torch.optim.Adam(hawkes_model.parameters(), lr = lr)
    hawkes_model.train()

    # Creating Penalty
    penalty = None
    # penalty = L2Penalty(weight = 0.02)

    pbar = tqdm(range(maxIterations), desc = "Optimizing Parameters")
    for step in pbar:
        optimizer.zero_grad()
        
        totalEvents = 0
        losses = []

        for seq in sequences:
            loss = hawkes_model.nll(seq)

            losses.append(loss)
            totalEvents += len(seq.mi)

        nllLoss = torch.stack(losses).sum()

        # Penalty Loss
        if penalty is not None:
            regLoss = penalty(hawkes_model.alpha)
        else:
            regLoss = torch.tensor(0.0, device=device)

        # Total Loss
        loss = nllLoss + regLoss

        loss.backward()
        optimizer.step()
        avgNLLEvent = torch.stack(losses).mean().item()

        if step % 5 == 0:
            pbar.set_postfix({
                "NLL/event": f"{avgNLLEvent:.4f}",
                "events": totalEvents
            })
        
    # Extract Parameters
    hawkes_model.eval()
    with torch.no_grad():
        tDummy = torch.tensor([0.0]).to(device)
        muValues = hawkes_model.mu(tDummy).cpu().numpy()
        alphaMatrix = hawkes_model.alpha.cpu().numpy()
        gammaValue = hawkes_model.gamma.cpu().numpy()

    # Reformat Alpha and Mu Values
    alphaMatrix = np.squeeze(alphaMatrix).T
    muValues = np.squeeze(muValues)

    # Alpha Smoothing
    epsilon = 0.3
    alphaMatrix = (1 - epsilon) * alphaMatrix + epsilon * (0.5 / M)

    eigenvalues = np.linalg.eigvals(alphaMatrix)
    maxEigenvalue = np.max(np.abs(eigenvalues))

    print(f"Maximum Eigenvalue: {maxEigenvalue:.4f}")
    
    # Print Summary
    print("")
    print("Final Model Parameters:")
    print(f"Alpha Matrix Shape: {alphaMatrix.shape}")
    print(f"Alpha Matrix Sum: {alphaMatrix.sum():.4f}")
    print(f"Average Baseline (Mu): {muValues.mean():.6f}")
    print(f"Decay Rate (Beta): {float(gammaValue):.2f}")
    
    return muValues, alphaMatrix, gammaValue, sequences


def shotResponsibility(dataFrames: list[pd.DataFrame], mu: np.ndarray, beta: int, alpha: np.ndarray, allGamePairs: dict, teamName: str, playerInformation: dict) -> tuple[dict, list]:
    """
    Function To Generate Responsibilities For Each Shot Using E Step In Hawkes Process For One Game.

    Parameters
    ----------
    dataFrame : list[pd.DataFrame],
        Non Scaled Data Frame For All Games.

    mu : np.ndarray
        Matrix Of Baseline Rates (Mu).

    beta : int
        Decay Value (Beta) Not Scaled.

    alpha : np.ndarray
        Matrix Of Alpha Values.

    allGamePairs : dict
        Returned From ProcessData().

    teamName : str
        Name Of Team Being Analyzed.

    playerInformation : dict
        Global Player Information Dictionary.

    Returns
    -------
    zAll : dict
        Dictionary Of Shot Responsibilities (Top 5 Values).

    fullShotList : list
        List Of Shots That Occurred.
    """

    # Creating Dictionary and Initialzing Time Variables
    zAll = []
    fullShotList = []

    # Loop over multiple games
    for dataFrame in dataFrames:

        Z = {}
        shotList = []

        times = dataFrame["Timestamp"].values

        # Collecting Pair Information
        pairsDict = allGamePairs[0]
        allPairs = list(pairsDict.keys())

        # Find Shot Event Indexes
        shotIndexes = dataFrame[dataFrame["Type"] == "Shot"].index
        goalNode = playerInformation[teamName][-1]["index"]
        goalPairs = [pair for pair in allPairs if pair[1] == goalNode]

        # Looping Over All Shots Shots
        for i in shotIndexes:
            shotList.append(i)
            initialTime = times[i]
            priorPasses = []
            expKernelValues = []

            # Keeping Track of Shooter Index
            shooterIndex = dataFrame.loc[i, "Sender"]
            goalIndex = dataFrame.loc[i, "Receiver"]
            shotPair = (shooterIndex, goalIndex)
            shotPairIndex = allPairs.index(shotPair)

            # Considering Prior Passes and Shots for Responsibility
            for j in range(max(0, i - 6), i):
                eventType = dataFrame.loc[j, "Type"]
                if dataFrame.loc[j, "Type"] in ["Pass", "Shot"]:
                    finalTime = times[j]
                    deltaT = abs(initialTime - finalTime)

                    # Keeping Track of Sender and Reciever Group
                    senderIndex = dataFrame.loc[j, "Sender"]
                    recieverIndex = dataFrame.loc[j, "Receiver"]
                    passPair = (senderIndex, recieverIndex)

                    # Determining Which Alpha to Use
                    if eventType == "Pass":
                        if passPair in allPairs:
                            passPairIndex = allPairs.index(passPair)
                            alphaValue = alpha[shotPairIndex, passPairIndex]
                        else:
                            alphaValue = alpha.min()
                    
                    # Rebound Alpha
                    else:
                        alphaValue = alpha.min()

                    # Calculating Exponential Kernel
                    expKernelValues.append(alphaValue * beta * np.exp((-1 * beta) * deltaT))
                    
                    # Updating Prior Passes
                    priorPasses.append(j)
            
            # Determing Mu Value 
            muValue = mu[shotPairIndex]

            # Determining Denominator Value
            denom = muValue + sum(expKernelValues)

            # Creating Shot Dictionary
            ZI = {}

            # Calculating Non Spontaneous Shot Probability
            for j, val in zip(priorPasses, expKernelValues):
                ZI[j] = val / denom

            # Calculating Spontaneous Shot Probability and Dropping
            baselineProbability = muValue / denom

            # Keep Top 5 Responsibilities
            sortedZI = sorted(ZI.items(), key = lambda x: x[1], reverse = True)
            ZI = dict(sortedZI[:6])

            # Normalizing Pass Responsibilities
            ZI = {k: round(float(v), 3) for k, v in ZI.items()}
            # Adding To Dictionary
            Z[i] = ZI

        zAll.append(Z)
        fullShotList.append(shotList)

    return zAll, fullShotList

