import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_plots():
    """
    Generate the plots for generated outputs
    """

    scores_df = pd.read_csv("output/autoencoder/scores.csv", header=0)
    plot_histogram(scores_df["cosine"].values, "Algorithm labelled data - all levels")

    pairs_df = pd.read_csv("data/confidence_intervals.csv", header=0, sep="\\s*,\\s*")
    pairs_df = validate_results(pairs_df, scores_df)
    plot_histogram(pairs_df["cosine"].values, "Algorithm labelled data - only for human labelled levels")

    plot_diff(pairs_df)


def plot_histogram(values, title):
    """
    Plot the histogram for supplied data

    Args:
        values: ndarray containing the scores to be plotted
        title: string title of the plot
    """

    plt.hist(values, bins=[1, 1.5, 2, 2.5, 3, 3.5, 4])
    plt.xlabel("Similarity score")
    plt.ylabel("Number of pairs")
    plt.title(title)
    plt.show()


def plot_diff(pairs_df):
    """
    Plot the difference between algorithmic score and human score

    Args:
        pairs_df: pandas dataframe containing score information
    """

    plt.plot(pairs_df["cosine"] - pairs_df["mean"])
    plt.plot([0] * pairs_df.shape[0])
    plt.xlabel("Pair ID")
    plt.ylabel("Score")
    plt.legend(["Algo score - Human score", "Zero line"])
    plt.show()


def validate_results(pairs_df, scores_df):
    """
    Fetch the corresponding algorithmic scores for the human assigned scores

    Args:
        pairs_df: pandas dataframe containing human assigned scores
        scores_df: pandas dataframe containing algorithmically generated scores

    Returns:
        pandas dataframe containing relevant columns
    """
    cosine = []
    for index, row in pairs_df.iterrows():
        left = int(row["left"])
        right = int(row["right"])

        val = scores_df[(scores_df["left"] == left) & (scores_df["right"] == right)]

        if not val.empty:
            cosine.append(val.iloc[0].cosine)
        else:
            val = scores_df[(scores_df["left"] == right) & (scores_df["right"] == left)]
            cosine.append(val.iloc[0].cosine)

    # Add columns to the dataframe
    pairs_df["cosine"] = np.array(cosine)

    return pairs_df
