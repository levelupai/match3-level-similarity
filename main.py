import argparse
import numpy as np

from autoencoder import dataset as auto_dataset, model, comparator, optimizer, generator, visualizer
from rule_based import dataset as rule_dataset, distance


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    help_msg = """
            Select the algorithm that you wish to execute. Options 1-5 are for the AutoEncoder only.
            1. Train the model
            2. Generate dense representation and level data
            3. Generate scores for all levels
            4. Optimize the weights for the vectors
            5. Generate the plots
            6. Rule-based algorithm
            """

    parser.add_argument("algorithm", help=help_msg, type=int)
    args = parser.parse_args()

    if args.algorithm == 1:
        # Train the model
        encoder = model.Encoder()
        encoder.train_model()
        encoder.test_model()

    elif args.algorithm == 2:
        # Get the dense representation and generate level data
        level_num = int(input("Enter level number: "))
        dense = auto_dataset.VectorDataset().get_dense_representation(level_num)

        print("\nDense representation of level {}: ".format(level_num))
        print(list(dense))

        # Get the dense representation and generate the level data
        gen = generator.LevelGenerator()
        level_str = gen.generate_level_data(dense)

        print("\nLevel data re-generated:")
        print(level_str)

    elif args.algorithm == 3:
        # Get the dataset
        level_vectors, labels, pairs_df = auto_dataset.VectorDataset(save_vectors=True).get_complete_data()

        # Run the comparator and compare all levels
        comp = comparator.Comparator(level_vectors, labels, pairs_df)
        comp.compare_all_levels(save=True, validate=True)

    elif args.algorithm == 4:
        # Get the dataset
        level_vectors, labels, pairs_df = auto_dataset.VectorDataset(save_vectors=True).get_complete_data()

        # Perform weight optimization for matrix
        optim = optimizer.WeightOptimizer(level_vectors, labels, pairs_df)
        weights = np.ones(level_vectors.shape[1])

        # Iterate continuously for converging the weights
        for i in range(10):
            print()
            print("-" * 40)
            print("\nIteration ", i)
            print()
            weights = optim.optimize_weights(weights, limit=30, step=0.25)

    elif args.algorithm == 5:
        # Generate the plots for the results
        visualizer.generate_plots()

    elif args.algorithm == 6:
        # Populate the dataset into memory
        levels_df, pairs_df = rule_dataset.AggregateDataset().get_complete_data()

        # Generate the cosine scores
        distance.generate_scores(levels_df, pairs_df, save=True, validate=True, visualize=True)


if __name__ == "__main__":
    main()
