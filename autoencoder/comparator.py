import numpy as np
import pandas as pd

from sklearn.metrics import pairwise

# Global constants
_BEST_WEIGHTS = np.array([30.25, 1, 15.5, 0.25, 1, 1, 1, 1, 1, 1])
_SCORES_PATH = "./output/autoencoder/scores.csv"

class Comparator(object):

    def __init__(self, level_vectors, labels, pairs_df):
        """
        Class for comparing all the vectors and generating similarity scores for them

        Args:
            level_vectors: ndarray containing dense vector representations of all levels
            labels: ndarray containing labels for respectively indexed vectors
            pairs_df: dataframe containing pair indices and respective mean scores
        """

        self.level_vectors = level_vectors
        self.labels = labels
        self.pairs_df = pairs_df

    def compare_all_levels(self, weights=_BEST_WEIGHTS, save=False, validate=False):
        """
        Generate cosine scores for all pairs of levels

        Returns:
            dataframe containing pairs of level ids and their scores
        """

        # Generate distances between each relevant levels pair
        scores = []
        for left in range(251):
            if left in self.labels:
                for right in range(left, 251):
                    if right in self.labels:
                        scores.append(self.compare_vectors(left, right, weights))

        # Create the dataframe
        scores_df = pd.DataFrame(scores, columns=["left", "right", "cosine"])

        # Normalize cosine scores
        distances = scores_df["cosine"].values
        distances = (((distances + 1) * (4 - 1)) / 2) + 1
        scores_df["cosine"] = distances

        # Sort the dataframe
        scores_df.sort_values("cosine", inplace=True, ascending=False)

        # Save the csv
        if save:
            self.save_df(scores_df)

        # Validate results
        if validate:
            self.validate_results(scores_df)

        return scores_df

    def compare_vectors(self, left, right, weights):
        """
        Compares the vectors for provided level numbers and generates cosine distance for them

        Args:
            left: Number of level 1
            right: Number of level 2
            weights: Weights vector for attributes

        Returns:
            List containing level 1 number, level 2 number, cosine distance
        """

        # Get the indices of the levels
        index_1 = self.labels == int(left)
        index_2 = self.labels == int(right)

        # Get the vectors
        vector_1 = self.level_vectors[index_1] * weights
        vector_2 = self.level_vectors[index_2] * weights

        # Generate the distances
        cosine = pairwise.cosine_similarity(vector_1, vector_2).item(0)

        return [left, right, cosine]

    def save_df(self, scores_df):
        """
        Save the supplied dataframe

        Args:
            scores_df: Dataframe to be saved
        """

        scores_df["left"] = scores_df["left"].values.astype(int)
        scores_df["right"] = scores_df["right"].values.astype(int)
        scores_df.to_csv(_SCORES_PATH, header=["left", "right", "cosine"], index=None)

    def validate_results(self, scores_df):
        """
        Validate whether the distances lie within a range of mean or not

        Args:
            scores_df: dataframe containing pairs information and their scores

        Returns:
            scores_df with additional columns about distances lying in or out of the intervals
        """

        cosine = []
        for index, row in self.pairs_df.iterrows():
            left = int(row["left"])
            right = int(row["right"])

            val = scores_df[(scores_df["left"] == left) & (scores_df["right"] == right)]

            if not val.empty:
                cosine.append(val.iloc[0].cosine)
            else:
                val = scores_df[(scores_df["left"] == right) & (scores_df["right"] == left)]
                cosine.append(val.iloc[0].cosine)

        cosine = np.array(cosine)

        # Get the intervals
        mean = self.pairs_df["mean"].values

        # Generate new columns for indicating validity of distances
        self.pairs_df["mean_match_1"] = np.where((cosine >= (mean - 1)) & (cosine <= (mean + 1)), True, False)
        self.pairs_df["mean_match_5"] = np.where((cosine >= (mean - 0.5)) & (cosine <= (mean + 0.5)), True, False)

        # Print the results
        print("\nNumber of values that lie within 1 score of mean: ", np.sum(self.pairs_df["mean_match_1"].values))
        print("Percentage: {}%".format(np.sum(self.pairs_df["mean_match_1"].values) / self.pairs_df.shape[0] * 100))

        print("\nNumber of values that lie within 0.5 score of mean: ", np.sum(self.pairs_df["mean_match_5"].values))
        print("Percentage: {}%".format(np.sum(self.pairs_df["mean_match_5"].values) / self.pairs_df.shape[0] * 100))
