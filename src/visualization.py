"""
Functions for Model Visulization

Author: Phillip Kornberg
Date: 9.22.26
"""

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from matplotlib import gridspec
from collections import Counter
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import seaborn as sns
import matplotlib as mpl

def frequencyOfPairs(passCounts: Counter) -> None:
    """
    Function To Visualize Frequency Of Passes Between Pairs To Determine Drop Off.

    Parameters
    ----------
    passCounts : Counter
        Dictionary Of Pairs And Pass Counts.
    """

    # Sorting Pass Counts
    frequencies = [v for _, v in passCounts.most_common()]

    # Ranking
    ranks = range(1, len(frequencies) + 1)
    
    # Plotting
    plt.figure(figsize=(12, 6), dpi = 300)
    plt.bar(ranks, frequencies, color="#154174")
    plt.xlabel("Pair Count")
    plt.ylabel("Frequency")
    # plt.title("Pair Frequencies")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

def alphaGoalClustermap(alpha: np.ndarray, teamName: str, playerInformation: dict, allGamePairs: list, threshold: float) -> tuple[np.ndarray, list]:
    """
    Function To Visualize Goal Alpha Values With Clustermap.

    Parameters
    ----------
    alpha : np.ndarray
        Alpha Values From Sparklen.

    teamName : str
        Name Of Team.

    playerInformation : dict
        Global Player Information.

    allGamePairs : list
        Dictionary Of Pairs And Timestamps.

    threshold : float
        Threshold for Alphas Displayed in Clustermap

    Returns
    -------
    alphaZoom : np.ndarray
        Matrix Of Goal Alphas.

    pairs : list
        List Of All Pair Keys.
    """

    # Creating Color Map
    blues = LinearSegmentedColormap.from_list("custom_blues",
        [
            (0.0, "#FFFFFF"),     
            (0.01, "#f3f8fc14"),  
            (0.20, "#6baed6"),  
            (0.50, "#2171b5"),   
            (1.00, "#08306b")   
        ])

    # Collecting Pairs to Create Row and Column Indexes
    games = allGamePairs[0]    
    goalNumber = playerInformation[teamName][-1]["index"]
    pairs = list(games.keys())
    
    rowPairs = [p for p in pairs if p[1] == goalNumber]
    rowIndexes = [pairs.index(p) for p in rowPairs]

    colIndexes = np.array([i for i, p in enumerate(pairs) if p[1] != goalNumber])

    # Selecting Portion of Alpha Matrix
    alphaZoom = alpha[np.ix_(rowIndexes, colIndexes)]

    # Creating Labels
    xLabels = [str(p) for p in pairs if p[1] != goalNumber]
    yLabels = [str(p) for p in rowPairs]

    # Create DataFrame From Goal Alphas
    df = pd.DataFrame(alphaZoom, index = yLabels, columns = xLabels)

    # Create Global Clustering
    g = sns.clustermap(
        df,
        method = "average",
        metric = "euclidean",
        cmap = blues,
        figsize = (20, 8),
        row_cluster = False,
        col_cluster = True,
        cbar = False,
        linewidths = 0.5,
        dendrogram_ratio = (0.06, 0.12),
    )

    colOrder = g.dendrogram_col.reordered_ind
    plt.close(g.fig)
    dfClustered = df.iloc[:, colOrder]

    # Keep Only Columns With Alpha > Threshold
    keepCols = (dfClustered > threshold).any(axis = 0)
    dfFiltered = dfClustered.loc[:, keepCols]

    # Creating Clustermap
    g = sns.clustermap(
        dfFiltered,
        method = "average",
        metric = "euclidean",
        cmap = blues,
        figsize = (20, 8),
        vmin = 0.00,
        xticklabels = dfFiltered.columns,
        yticklabels = yLabels,
        row_cluster = False,
        col_cluster = False,  
        cbar_kws = {"orientation": "vertical"},
        cbar_pos = (0.02, 0.15, 0.01, 0.7),
        linewidths = 0.5,
        dendrogram_ratio = (0.06, 0.12),
    )


    g.ax_heatmap.set_title(f"Clustermap of {teamName} Goal Alphas", pad = 60, fontsize = 20)
    g.ax_heatmap.set_xlabel("All Pairs", fontsize = 15)
    g.ax_heatmap.set_ylabel("\nGoal Pairs", fontsize = 15)

    g.cax.set_title("Alphas", fontsize = 15, pad = 10)

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation = 45, fontsize = 10, ha = "right")
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize = 10)
    g.ax_cbar.tick_params(labelsize = 10)
    plt.show()

    return alphaZoom, pairs

def hawkesHeatmap(mu: np.ndarray, alpha: np.ndarray) -> None:
    """
    Function To Plot Mu And Alpha Values Generated.

    Parameters
    ----------
    mu : np.ndarray
        Mu Array From Sparklen.

    alpha : np.ndarray
        Alpha Values From Sparklen.
    """

    # Plotting Info
    plt.rcParams.update({
    'font.size': 6,          
    'axes.titlesize': 7,    
    'axes.labelsize': 6,    
    'xtick.labelsize': 5,   
    'ytick.labelsize': 5,   
    'legend.fontsize': 6,
    })

    # Initilizing Figure
    fig = plt.figure(figsize = (7, 5), constrained_layout = True)

    gs = gridspec.GridSpec(
        1, 4, figure = fig,
        width_ratios = [1.2, 20, 1, 1] 
    )

    # Creating Mu Vector
    axMu = fig.add_subplot(gs[0])

    imMu = axMu.imshow(mu[:, np.newaxis], aspect = 'auto', cmap = 'Reds')
    axMu.set_xticks([])
    axMu.set_title("μ")
    
    axMu.yaxis.tick_left()
    axMu.yaxis.set_label_position("left")
    axMu.set_ylabel("Pair")

    # Creating Box Around Mu
    for spine in axMu.spines.values():
        spine.set_visible(False)

    # Creating Alpha Matrix
    axAlpha = fig.add_subplot(gs[1], sharey = axMu)
    imAlpha = axAlpha.imshow(alpha, aspect = 'auto', cmap = 'Blues')
    axAlpha.set_title("α")
    axAlpha.set_xlabel("Pair")

    axAlpha.tick_params(axis = 'y', labelleft = False)

    # Drawing Box Around ALpha
    for spine in axAlpha.spines.values():
        spine.set_visible(False)

    # Setting Colorbars
    caxMu = fig.add_subplot(gs[2])
    cbarMu = fig.colorbar(imMu, cax = caxMu)
    cbarMu.outline.set_visible(False)

    cbarMu.formatter = mticker.FormatStrFormatter('%.6f')
    cbarMu.update_ticks()

    caxAlpha = fig.add_subplot(gs[3])
    cbarAlpha = fig.colorbar(imAlpha, cax = caxAlpha)
    cbarAlpha.outline.set_visible(False)

    # Drawing Plot
    plt.show()

def createPassingNetwork(teamName: str, gameDfs: list[pd.DataFrame], passThreshold: int, playerInformation: dict, margin: int, k: float) -> None:
    """
    Function To Build Season Long Passing Network.

    Parameters
    ----------
    teamName : str
        Name Of Team.

    gameDfs : list[pd.DataFrame]
        Array Of Game DataFrames From MultiGameCollection().

    passThreshold : int
        Minimum Pass Count For Displaying Edge.

    playerInformation : dict
        Dictionary Returned From MultiGameCollection().

    margin : int
        Spacing Between Edges

    k : float
        Node Spacing
    """

    # Creating Directed Graph
    DG = nx.MultiDiGraph()
    data = pd.concat(gameDfs, ignore_index = True)

    # Counting Passes for Filtering
    passCounts = Counter(zip(data["Sender"], data["Receiver"]))

    # Adding Edges
    for (sender, receiver), weight in passCounts.items():
        DG.add_edge(sender, receiver, weight = weight)

    # Collecting Player Last Names
    teamInfromation = playerInformation[teamName]
    lastnames = {
        details['index']: (
            'Goal' if details['name'] == 'Goal Node'
            else 'Messi' if details['name'] == 'Lionel Andrés Messi Cuccittini'
            else "Javier" if details['name'] == "Javier Alejandro Mascherano"
            else details['name'].split()[-1]
        )
        for pid, details in teamInfromation.items()
    }

    # Labeling and Filtering Graph
    DGLabeled = nx.relabel_nodes(DG, lastnames)

    DGFiltered = nx.MultiDiGraph()
    for sender, receiver, d in DGLabeled.edges(data = True):
        if d['weight'] >= passThreshold:
            DGFiltered.add_edge(sender, receiver, weight = d['weight'])
    isolated = list(nx.isolates(DGFiltered))
    DGFiltered.remove_nodes_from(isolated)
    if "Goal" in DGFiltered:
        DGFiltered.remove_node("Goal")

    # Determing Graph Position and Edges
    pos = nx.spring_layout(DGFiltered, k, iterations = 300, seed = 42)
    center = np.mean(list(pos.values()), axis = 0)
    pos = {node: (coord - center) * 1.5 + center for node, coord in pos.items()}

    edges = list(DGFiltered.edges(data = True))

    # Sorting Edges by Weight
    edges_sorted = sorted(edges, key = lambda x: x[2]['weight'])
    weights = [d['weight'] for _, _, d in edges_sorted]
    max_w = max(weights)
    widths = [3 * w / max_w for w in weights]

    # Plotting Figure
    fig, ax = plt.subplots()

    node_colors = [
        'lightskyblue'
        for node in DGFiltered.nodes()
    ]

    # Drawing Graph
    nx.draw_networkx_nodes(DGFiltered, pos, node_color = node_colors, edgecolors = 'black', node_size = 1500, ax = ax)
    nx.draw_networkx_labels(DGFiltered, pos, ax = ax, font_size = 7)
    nx.draw_networkx_edges(
        DGFiltered, pos,
        edgelist = [(u, v) for u, v, _ in edges_sorted],
        width = widths,
        edge_color = weights,
        edge_cmap = plt.cm.Blues,
        arrows = True,
        arrowsize = 15,
        connectionstyle = 'arc3,rad=0.1',
        min_target_margin = margin,
        ax = ax
    )

    sm = plt.cm.ScalarMappable(cmap = plt.cm.Blues, norm = plt.Normalize(vmin = min(weights), vmax = max_w))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax = ax, shrink = 0.5)
    cbar.ax.tick_params(labelsize = 6)
    cbar.ax.set_title("Pass Count", pad = 8, fontsize = 6)

    plt.title(f"{teamName} 2015/2016 Passing Network", fontsize = 8)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def playerXGIHeatmap(dataFrames: list[pd.DataFrame], allResponsibilities: list[dict], playerName: str,
    playerID: int, teamName: str, playerInformation: dict) -> None:
    """
    Function To Display Player xGI Heatmap.

    Parameters
    ----------
    dataFrames : list[pd.DataFrame]
        List Of Game DataFrames.

    allResponsibilities : list[dict]
        Array Of All Shot Responsibilities.

    playerName : str
        Preferred Player Name.

    playerID : int
        Player ID Number.

    teamName : str
        Name Of Team.

    playerInformation : dict
        Dictionary Of Player Information.
    """

    # Collecting Player Information
    index = playerInformation[teamName][playerID]["index"]
    name = playerInformation[teamName][playerID]["name"]

    # Field Dimensinos
    width, height = 120, 80
    cellSize = 4 
    gridW = int(np.ceil(width / cellSize))
    gridH = int(np.ceil(height / cellSize))

    grid = np.zeros((gridH, gridW))
    total_xgi = 0

    # Looping Through Each Game
    for gameIndex, gameDf in enumerate(dataFrames):

        responsibilities = allResponsibilities[gameIndex]

        # Build playerRows BEFORE any index reset
        playerDf = gameDf[gameDf["Sender"] == index]
        playerRows = set(playerDf.index)

        # Build xgiMap using original indices
        xgiMap = {}
        for shotRow, contributions in responsibilities.items():
            xG = gameDf.loc[shotRow, "xG"]
            for eventID, weight in contributions.items():
                if eventID == -1:
                    continue
                if eventID in playerRows:
                    xgiMap[eventID] = xgiMap.get(eventID, 0) + round(weight * xG, 3)

        # Now work with player's rows
        df = playerDf[playerDf.index.isin(xgiMap.keys())].copy()
        df['x'] = df['Location'].apply(lambda loc: loc[0])
        df['y'] = df['Location'].apply(lambda loc: loc[1])

        for idx, row in df.iterrows():
            gx = min(int(row['x'] // cellSize), gridW - 1)
            gy = min(int(row['y'] // cellSize), gridH - 1)
            weight = xgiMap.get(idx, 0)
            grid[gy, gx] += weight
            total_xgi += weight

    # Normalzining Grid
    norm_grid = np.power(grid / grid.max(), 0.7) if grid.max() > 0 else grid

    # Setting Heatmap Colors
    heatColors = [
        (1.0, 1.0, 1.0, 0.0),    
        (0.85, 0.92, 1.0, 0.5),  
        (0.5, 0.75, 1.0, 0.7),   
        (0.2, 0.5, 0.9, 0.85),   
        (0.05, 0.2, 0.6, 1.0)    
    ]
    heatCmap = LinearSegmentedColormap.from_list('tactical_blue', heatColors)

    # Plotting
    fig, ax = plt.subplots(figsize = (14, 10))
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-5, width + 5)
    ax.set_ylim(-5, height + 5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Draw heatmap
    for y in range(gridH):
        for x in range(gridW):
            val = norm_grid[y, x]
            if val > 0:
                rect_color = heatCmap(val)
                ax.add_patch(patches.Rectangle(
                    (x * cellSize, y * cellSize),
                    cellSize, cellSize,
                    facecolor=rect_color,
                    edgecolor='white',
                    linewidth=0.5,
                    zorder=2
                ))

    # Setting Picth Width
    LINE_KW = dict(color = "#000000", linewidth = 2.2, zorder = 3)

    # Main Field
    ax.add_patch(patches.Rectangle((0, 0), width, height, fill = False, **LINE_KW))
    ax.plot([width/2, width/2], [0, height], **LINE_KW) 
    ax.add_patch(patches.Circle((width/2, height/2), 9.15, fill = False, **LINE_KW)) 
    ax.scatter(width/2, height/2, color = 'black', s = 30, zorder = 4) 

    # Left Penalty Areas
    ax.add_patch(patches.Rectangle((0, height/2-20.15), 16.5, 40.3, fill = False, **LINE_KW))
    ax.add_patch(patches.Rectangle((0, height/2-9.15), 5.5, 18.3, fill = False, **LINE_KW))
    ax.add_patch(patches.Arc((11, height/2), 18.3, 18.3, theta1 = 308, theta2 = 52, **LINE_KW))

    # Right Penalty Areas
    ax.add_patch(patches.Rectangle((width-16.5, height/2-20.15), 16.5, 40.3, fill = False, **LINE_KW))
    ax.add_patch(patches.Rectangle((width-5.5, height/2-9.15), 5.5, 18.3, fill = False, **LINE_KW))
    ax.add_patch(patches.Arc((width-11, height/2), 18.3, 18.3, theta1 = 128, theta2 = 232, **LINE_KW))

    # Drawing Corner Kicks
    corner_details = [(0,0,0,90), (0,80,270,360), (120,0,90,180), (120,80,180,270)]
    for cx, cy, t1, t2 in corner_details:
        ax.add_patch(patches.Arc((cx, cy), 4, 4, theta1 = t1, theta2 = t2, **LINE_KW))

    # Drawing Titleand Subtitle
    ax.text(width / 2, height + 6, f'{playerName} xGI Heatmap', fontsize = 34, ha = 'center')
    ax.text(width / 2, height + 2, f'{total_xgi:.2f} xGI', 
            fontsize = 14, color = "#000000", ha = 'center')
    
    plt.show()

def influenceGraph(playerInformation: dict, alpha: np.ndarray, originPass: tuple, influencing: bool, pairs: list, teamName: str, threshold : float) -> None:
    """
    Function To Display Passes That Infuence or Are Influenced by Origin Pass.  

    Parameters
    ----------
    playerInformation : dict
        Dictionary Of Player Information.

    alpha: np.ndarray
        Alpha Matrix.

    originPass : tuple
        Starting Pass.

    influencing: bool
        True if Want to Find Passes Influencing Origin Pass, False if Want to Find Passes Influenced by Origin Pass 

    pairs: list
        List of All Pairs in Alpha Matrix

    teamName: str
        Name of Team

    threshold : float
        Threshold for Alpha Value
    """

    # Setting Alpha Threshold and Row Initialization
    rows = []

    # Looping Through Alpha Matrix To Find Pairs
    for i in range(alpha.shape[0]):
        rowPair = pairs[i]
        activeIndexes = np.where(alpha[i] > threshold)[0]

        # Extracting Important Alpha Values
        for j in activeIndexes:
            colPair = pairs[j]
            val = alpha[i, j]
            rows.append({"rowPair": rowPair, "colPair": colPair, "alphaValue": val})

    # Creating DataFrame and Filtering
    df = pd.DataFrame(rows)
    if influencing:
        df = df[df["rowPair"] == originPass]
        df = df.drop(columns = "rowPair")
        df['sender'] = df['colPair'].str[0]
        df['reciever'] = df['colPair'].str[1]

        # Removing Goal Passes From Influencing Network
        df = df[df['reciever'] != 23]

    else:
        df = df[df["colPair"] == originPass]
        df = df.drop(columns = "colPair")
        df['sender'] = df['rowPair'].str[0]
        df['reciever'] = df['rowPair'].str[1]

    # Creating Directed Graph
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['sender'], row['reciever'], weight = row['alphaValue'])

    # Extracting Last Names
    teamInformation = playerInformation[teamName]
    lastnames = {
            details['index']: (
                'Goal' if details['name'] == 'Goal Node'
                else 'Messi' if details['name'] == 'Lionel Andrés Messi Cuccittini'
                else details['name'].split()[-1]
            )
            for pid, details in teamInformation.items()
        }

    # Adding Origin Pass For Visualization
    originSender = lastnames[originPass[0]]
    originReceiver = lastnames[originPass[1]]
    G.add_edge(originSender, originReceiver, weight = 0)

    # Relabeling Graph
    G = nx.relabel_nodes(G, lastnames)
    G.remove_nodes_from(list(nx.isolates(G)))

    # Sorting Edges by Weight For Better Visualization
    edges = list(G.edges(data = True))
    edges_sorted = sorted(edges, key = lambda x: x[2]['weight'])
    weights = [d['weight'] for _, _, d in edges_sorted]

    # Updating Colormap So Weak Alphas Are Still Visible
    def truncate_colormap(cmap, minval = 0.2, maxval = 1.0, n = 256):
        return mpl.colors.LinearSegmentedColormap.from_list(
            f"trunc_{cmap.name}",
            cmap(np.linspace(minval, maxval, n))
        )
    blues = truncate_colormap(plt.cm.Blues, 0.2, 1.0)

    # Plotting Graph
    if not weights:
        print("No edges to plot")
    else:
        pos = nx.spring_layout(G, seed = 42, k = 0.8)
        widths = 1.5
        fig, ax = plt.subplots()

        # Drawing Graph
        nx.draw_networkx_nodes(G, pos, node_size = 650, node_color = "lightskyblue", edgecolors = "black")
        nx.draw_networkx_labels(G, pos, font_size = 4.5)

        # Drawing Normal Edges
        nx.draw_networkx_edges(
            G, pos,
            edgelist = [(u, v) for u, v, _ in edges_sorted if (u, v) != (originSender, originReceiver)],
            width = widths,
            edge_color = weights[:-1] if (originSender, originReceiver) in [(u, v) for u, v, _ in edges_sorted] else weights,
            edge_cmap = blues,
            arrows = True,
            arrowsize = 15,
            min_target_margin = 18,
            connectionstyle = "arc3,rad=0.1"
        )

        # Finding Origin Pass to Graph
        originEdges = [(originSender, originReceiver)]

        # Drawing Origin Pass in Red
        if originEdges:
            nx.draw_networkx_edges(
                G, pos,
                edgelist = originEdges,
                width = 1.5,
                edge_color = "red",
                arrows = True,
                arrowsize = 15,
                min_target_margin = 18,
                connectionstyle = "arc3,rad=0.1"
            )

        # Drawing Colorbar
        sm = plt.cm.ScalarMappable(
            cmap = blues,
            norm = plt.Normalize(vmin = min(weights), vmax = max(weights))
        )
        cbar = plt.colorbar(sm, ax = ax, shrink = 0.5)
        cbar.ax.set_title("Alpha", pad = 8, fontsize = 8)
        
        # Writing Title
        sender = lastnames[originPass[0]]
        reciever = lastnames[originPass[1]]

        if influencing:
            plt.title(f"Passes Influencing {sender} to {reciever} Interaction", fontsize = 10)
        else:
            plt.title(f"Passes Influenced by {sender} to {reciever} Pass", fontsize = 10)

        plt.axis("off")
        plt.show()