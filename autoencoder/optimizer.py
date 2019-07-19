import numpy as np

from autoencoder import comparator


class WeightOptimizer(object):

    def __init__(self, level_vectors, labels, pairs_df):
        """
        Class for optimizing the weights vector for MTS level analysis

        Args:
            level_vectors: ndarray containing dense vector representations of all levels
            labels: ndarray containing labels for respectively indexed vectors
            pairs_df: dataframe containing pair indices and respective mean scores
        """

        # Initialize the data
        self.level_vectors = level_vectors
        self.labels = labels
        self.pairs_df = pairs_df

        self.weights = np.ones(level_vectors.shape[1])

        # Initialize required data structures for keeping track of the best answer
        self.best_weights = np.ones(level_vectors.shape[1])
        self.least_diff = -1

        # Initialize the comparator
        self.comparator = comparator.Comparator(self.level_vectors, self.labels, self.pairs_df)

    def optimize_weights(self, weights, limit=10, step=1):
        """
        Find the optimal weights through greedy search

        Args:
            weights: ndarray of initial weights
            limit: maximum range of weights above and below the original weight
            step: step size for varying the weights

        Returns:
            ndarray containing best weights
        """

        print("Optimizing weights\n")

        best_score = -1
        self.weights = weights
        for i in range(len(weights)):
            og_wt = weights[i]
            best_wt = og_wt

            while weights[i] < og_wt + limit:
                # Generate the cosine scores for all the vectors
                scores_df = self.comparator.compare_all_levels(weights, save=False, validate=False)

                # Get the analysis
                score = self.analyze_scores(scores_df)

                if best_score == -1 or score > best_score:
                    print("Weights updated: ", weights)
                    print("Best score: ", score)
                    print()
                    best_score = score
                    best_wt = weights[i]

                weights[i] += step

            weights[i] = og_wt
            while weights[i] > og_wt - limit:
                # Generate the cosine scores for all the vectors
                scores_df = self.comparator.compare_all_levels(weights, save=False, validate=False)

                # Get the analysis
                score = self.analyze_scores(scores_df)

                if best_score == -1 or score > best_score:
                    print("Weights updated: ", weights)
                    print("Best score: ", score)
                    print()
                    best_score = score
                    best_wt = weights[i]

                weights[i] -= step

            weights[i] = best_wt

        return weights

    def analyze_scores(self, scores_df):
        """
        Analyze the scores to check if they have been optimized or not

        Args:
            scores_df: dataframe containing scores information for all pairs

        Returns:
            percentage of scores that lie within 0.5 range of the human scores
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
        mean = self.pairs_df["mean"].values

        values = np.where((cosine >= (mean - 1.0)) & (cosine <= (mean + 1.0)), True, False)
        score = np.sum(values) / values.shape[0]

        return score
