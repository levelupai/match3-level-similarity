import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise

# Global constants
_BEST_WEIGHTS = np.array([57.0, 10, 48.5, 25.5, 69.5, 38.0, 20, 51.0, 26.5, 48.5, 10, 20, 20])
_SCORES_PATH = "./output/rule_based/scores.csv"


def generate_scores(levels_df, pairs_df, save=True, validate=True, visualize=True):
    """
    Generates the cosine similarity scores for all possible pairs of aggregated level vectors

    Args:
        levels_df: pandas dataframe containing aggregated level vectors
        pairs_df: pandas dataframe containing human ratings and pairs information
        save: boolean to save the resulting scores or not
        validate: boolean to validate the generated scores or not
        visualize: boolean to visualize the plots of the results or not
    """

    # Initialize the weights
    weights = _BEST_WEIGHTS

    # Normalize the aggregated vectors
    levels_df = normalize_df(levels_df)

    # Generate the cosine scores for all the pairs
    scores = []
    for left in range(1, 251):
        for right in range(left, 251):
            try:
                left_vec = levels_df.loc[left].values.reshape(1, -1) * weights
                right_vec = levels_df.loc[right].values.reshape(1, -1) * weights

                cos_dist = pairwise.cosine_similarity(left_vec, right_vec)

                scores.append((left, right, cos_dist[0, 0]))
            except KeyError:
                pass

    # Create the dataframe
    scores_df = pd.DataFrame(scores, columns=["left", "right", "cosine"])

    # Rescale cosine scores
    cosine = scores_df.cosine.values
    cosine = (((cosine - 0) * (4 - 1)) / 1) + 1
    scores_df["cosine"] = cosine

    # Sort the dataframe
    scores_df.sort_values("cosine", inplace=True, ascending=False)

    # Save the csv
    if save:
        save_df(scores_df)

    # Add validation columns to the dataframe
    pairs_df = add_validation_columns(pairs_df, scores_df)

    # Validate the dataframe
    if validate:
        # Print the results
        print()
        print("-" * 40)
        print()

        print("\n\nNumber of values that lie within 1 score of mean: ", np.sum(pairs_df["mean_match_1"].values))
        print("Percentage: {}%".format(np.sum(pairs_df["mean_match_1"].values) / pairs_df.shape[0]))

        print("\nNumber of values that lie within 0.5 score of mean: ", np.sum(pairs_df["mean_match_5"].values))
        print("Percentage: {}%".format(np.sum(pairs_df["mean_match_5"].values) / pairs_df.shape[0]))

    # Plot the dataframe and visualize the distributions
    if visualize:
        plot_df(pairs_df["mean"].values, "Human labelled data")
        plot_df(scores_df["cosine"].values, "Algorithm labelled data - all levels")
        plot_df(pairs_df["cosine"].values, "Algorithm labelled data - only for human labelled levels")


def normalize_df(levels_df):
    """
    Normalize the passed dataframe

    Args:
        levels_df: pandas dataframe to be normalized

    Returns:
        Normalized dataframe
    """

    result = levels_df.copy()

    # Normalize each column one by one
    for feature_name in levels_df.columns:
        max_value = levels_df[feature_name].max()
        min_value = levels_df[feature_name].min()
        result[feature_name] = (levels_df[feature_name] - min_value) / (max_value - min_value)

    return result


def save_df(scores_df):
    """
    Save the dataframe to the global path

    Args:
        scores_df: dataframe to be saved
    """

    scores_df["left"] = scores_df["left"].values.astype(int)
    scores_df["right"] = scores_df["right"].values.astype(int)
    scores_df.to_csv(_SCORES_PATH, header=["left", "right", "cosine"], index=None)


def add_validation_columns(pairs_df, scores_df):
    """
    Adds the relevant generated scores of pairs rated by humans and information about the generated scores
    with reference to human ratings

    Args:
        pairs_df: pandas dataframe containing human ratings and pairs information
        scores_df: pandas dataframe containing generated ratings of all possible pairs

    Returns:
        pairs_df dataframe with additional columns
    """

    # Get generated scores for pairs rated by humans
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

    # Cast list to numpy
    cosine = np.array(cosine)

    # Add columns to the dataframe
    pairs_df["cosine"] = cosine

    # Get the intervals
    mean = pairs_df["mean"].values

    # Generate new columns for indicating validity of distances
    pairs_df["mean_match_1"] = np.where((cosine >= (mean - 1)) & (cosine <= (mean + 1)), True, False)
    pairs_df["mean_match_5"] = np.where((cosine >= (mean - 0.5)) & (cosine <= (mean + 0.5)), True, False)

    return pairs_df


def plot_df(values, title):
    """
    Plot the passed score distribution as a histogram

    Args:
        values: ndarray containing scores
        title: string title of the plot
    """

    plt.hist(values, bins=[1, 1.5, 2, 2.5, 3, 3.5, 4])
    plt.xlabel("Similarity score")
    plt.ylabel("Number of pairs")
    plt.title(title)
    plt.show()
