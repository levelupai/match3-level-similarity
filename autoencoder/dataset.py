import glob
import importlib
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from autoencoder import network

# Global constants for the data
_CHANNELS = 35
_MAX_BOARD_ROWS = 12
_MAX_BOARD_COLS = 12
_LEVELS_DATA_PATH = "./data/level/*.py"
_PAIRS_PATH = "./data/confidence_intervals.csv"
_MODEL_NAME = "./output/autoencoder/model.pth"
_LATENT_VECTORS_PATH = "./output/autoencoder/latent_vectors.csv"


class LevelsDataset(object):

    def __init__(self):
        """
        Class for reading and parsing the levels data and compiling the dataset as a numpy array
        """

        self.levels, self.labels = self.read_level_data()

    def get_data(self):
        """
        Returns the levels dataset as a numpy array
        """

        return self.levels

    def get_labels(self):
        """
        Returns the labels dataset as a numpy array
        """

        return self.labels

    def read_level_data(self):
        """
        Reads all the level files one file at a time

        Returns:
            levels dataset as a numpy array
        """

        # Open the directory
        files = glob.glob(_LEVELS_DATA_PATH)

        # The levels data
        levels = []

        # The labels
        labels = []

        # Read all the files in the directory one by one
        for file in files:
            index = self.parse_label_data(file)

            # Skip special case levels
            if index > 250:
                continue

            level = importlib.import_module("data.level.level%d" % index)

            levels.append(self.parse_level_data(level.data))
            labels.append(index)

        return np.array(levels), np.array(labels)

    def parse_level_data(self, json_data):
        """
        Parses the JSON data of a level and builds the required numpy array

        Args:
            json_data: JSON formatted data of a level

        Returns:
            numpy ndarray of level data
        """

        # A default matrix for level data
        level_data = np.zeros(shape=(_CHANNELS, _MAX_BOARD_ROWS, _MAX_BOARD_COLS))

        # Get the data for pieces on each cell
        for coords in json_data["board_info"]:
            coord_data = json_data["board_info"][coords]

            x, y = coords

            if 0 <= x < _MAX_BOARD_ROWS and 0 <= y < _MAX_BOARD_COLS:
                for key in coord_data:
                    # Add basic shape
                    level_data[0][x][y] = 1

                    # Check for base elements
                    if key == "base":
                        value = coord_data[key][0]

                        if value <= 6:
                            level_data[value][x][y] = 1
                        elif 10 <= value <= 14:
                            level_data[value - 3][x][y] = 1
                        elif 50 <= value <= 52:
                            level_data[value - 36][x][y] = 1
                        elif value == 55:
                            level_data[18][x][y] = 1
                        elif value == 66:
                            level_data[30 + coord_data[key][1]][x][y] = 1

                    # Check for cover elements
                    elif key == "cover":
                        value = coord_data[key][0]
                        level = coord_data[key][1]

                        if value == 60:
                            level_data[18 + level][x][y] = 1
                        elif value == 61:
                            level_data[21 + level][x][y] = 1
                        elif value == 63:
                            level_data[23 + level][x][y] = 1
                        elif value == 64:
                            level_data[25 + level][x][y] = 1
                        elif value == 66:
                            level_data[30 + level][x][y] = 1

                    # Check for background elements
                    elif key == "bg_number" or key == "background_number":
                        value = coord_data[key]
                        level_data[value - 28][x][y] = 1

        level_data[17] = self.check_apple_box(json_data)
        level_data[34] = self.check_wall_info(json_data)

        return level_data

    def parse_label_data(self, label):
        """
        Returns the level number from the filename

        Args:
            label: label of the vector
        """

        return int(label.split("/")[-1].replace("level", "").replace(".py", ""))

    def check_apple_box(self, json_data):
        """
        Builds a numpy array of information about apple boxes on game board

        Args:
            json_data: JSON formatted data of a level

        Returns:
              numpy array containing apple box location information
        """

        data = np.zeros((_MAX_BOARD_ROWS, _MAX_BOARD_COLS))

        if "apple_box_info" in json_data:
            for coords in json_data["apple_box_info"]:
                if coords[0] < _MAX_BOARD_ROWS and coords[1] < _MAX_BOARD_COLS:
                    data[coords[0]][coords[1]] = 1

        return data

    def check_wall_info(self, json_data):
        """
        Builds a numpy array of information about walls on game board

        Args:
            json_data: JSON formatted data of a level

        Returns:
              numpy array containing walls location information
        """

        data = np.zeros((_MAX_BOARD_ROWS, _MAX_BOARD_COLS))

        if "wall_info" in json_data:
            for coords in json_data["wall_info"]:
                if coords[0][0] < _MAX_BOARD_ROWS and coords[0][1] < _MAX_BOARD_COLS:
                    data[coords[0][0]][coords[0][1]] = 1
                if coords[1][0] < _MAX_BOARD_ROWS and coords[1][1] < _MAX_BOARD_COLS:
                    data[coords[1][0]][coords[1][1]] = 1

        return data


class VectorDataset(object):

    def __init__(self, save_vectors=False):
        """
        Class for instantiating vector data by passing original data through the PyTorch model
        """

        self.level_vectors, self.labels = self.get_vectors()
        self.pairs_df = self.read_human_labels()

        if save_vectors:
            self.save_vectors()

    def get_vectors(self):
        """
        Generate vector representation of original level data by passing it to the model

        Returns:
            ndarray of vectors and corresponding labels
        """

        # Setup the model for getting encoded vectors of level data
        model = network.AutoEncoderNetwork().double()
        model.load_state_dict(torch.load(_MODEL_NAME))
        model.set_encoder_needed(True)
        model.set_decoder_needed(False)
        model.eval()

        # Get the complete dataset
        dataset_builder = LevelsDataset()
        full_dataset = dataset_builder.get_data()
        labels = dataset_builder.get_labels()

        dataloader = torch.utils.data.DataLoader(full_dataset, batch_size=64, num_workers=0)

        vectors = []
        for level_data in dataloader:
            outputs = model(level_data)

            # Add the vectors into the list
            np_op = outputs.detach().numpy()
            for op in np_op:
                vectors.append(op)

        return np.array(vectors), labels

    def read_human_labels(self):
        """
        Reads confidence intervals and saves the pair data

        Returns:
            pairs dataset as a pandas dataframe
        """

        pair_df = pd.read_csv(_PAIRS_PATH, header=0, sep="\\s*,\\s*", usecols=["left", "right", "mean", "std_dev"])
        return pair_df

    def save_vectors(self):
        """
        Saves the latent space vectors to a CSV file
        """

        vector_df = pd.DataFrame(self.level_vectors)
        vector_df["labels"] = self.labels
        vector_df = vector_df.set_index("labels")
        vector_df = vector_df.sort_index()

        vector_df.to_csv(_LATENT_VECTORS_PATH, header=True)

    def get_dense_representation(self, num):
        """
        Generates the dense representation for provided level number

        Args:
            num: Number of level

        Returns:
            dense representation
        """

        # Get the indices of the levels
        index = self.labels == int(num)

        # Get the dense representation
        dense = self.level_vectors[index]

        return dense

    def get_complete_data(self):
        """
        Returns all the data generated including vectors, labels and pairs data
        """

        return self.level_vectors, self.labels, self.pairs_df
