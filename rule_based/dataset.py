import glob
import importlib
import pandas as pd

# Global constants for the data
_LEVELS_DATA_PATH = "./data/level/*.py"
_PAIRS_PATH = "./data/confidence_intervals.csv"


class AggregateDataset(object):

    def __init__(self):
        """
        Class for reading and parsing the levels data and compiling the dataset as a pandas dataframe
        """

        self.levels_df = self.read_level_data()
        self.pairs_df = self.read_pair_data()

    def get_complete_data(self):
        """
        Returns all the data generated including aggregate vectors and pairs data
        """

        return self.levels_df, self.pairs_df

    def read_level_data(self):
        """
        Reads all the level files one file at a time

        Returns:
            levels dataset as a pandas dataframe
        """

        # Open the directory
        files = glob.glob(_LEVELS_DATA_PATH)

        # The levels data
        levels = []

        # Read all the files in the directory one by one
        for file in files:
            index = self.parse_label_data(file)

            # Skip special case levels
            if index > 250:
                continue

            level = importlib.import_module("data.level.level%d" % index)
            levels.append(self.parse_level_data(level.data))

        levels_df = pd.DataFrame(levels)
        levels_df = levels_df.set_index("level_index").sort_index()

        return levels_df

    def parse_level_data(self, json_data):
        """
        Parses and aggregates the level's JSON data and puts the required data in a dictionary

        Args:
            json_data: The JSON data for a particular level

        Returns:
            A dictionary containing required level data
        """

        # A default matrix for level data
        level = {"level_index": 0, "powerups": 0, "background": 0, "cherry": 0, "donut": 0, "foam": 0,
                 "box": 0, "ice": 0, "chain": 0, "jelly": 0, "cookie": 0, "moves": 0, "walls": 0, "apple_box": 0}

        # Get the data for pieces on each cell
        for coords in json_data["board_info"]:
            coord_data = json_data["board_info"][coords]

            # Populate the level data
            level["level_index"] = json_data["level_index"]
            level["moves"] = json_data["move_count"]

            # Aggregate and populate the level data
            for key in coord_data:
                if key == "base":
                    piece_val = coord_data[key][0]
                    if 10 <= piece_val <= 14:
                        level["powerups"] += 1
                    elif piece_val == 50:
                        level["cherry"] += 1
                    elif piece_val == 51:
                        level["donut"] += 1
                    elif piece_val == 52:
                        level["foam"] += 1
                    elif piece_val == 66:
                        level["cookie"] += 1

                elif key == "cover":
                    piece_val = coord_data[key][0]
                    if piece_val == 60:
                        level["box"] += 1
                    elif piece_val == 61:
                        level["ice"] += 1
                    elif piece_val == 63:
                        level["chain"] += 1
                    elif piece_val == 64:
                        level["jelly"] += 1

                elif key == "bg_number" or key == "background_number":
                    level["background"] += 1

        if "wall_info" in json_data:
            for walls in json_data["wall_info"]:
                level["walls"] += len(walls)

        if "apple_box_info" in json_data:
            level["apple_box"] += len(json_data["apple_box_info"])

        return level

    def parse_label_data(self, label):
        """
        Returns the level number from the filename

        Args:
            label: label of the vector
        """

        return int(label.split("/")[-1].replace("level", "").replace(".py", ""))

    def read_pair_data(self):
        """
        Reads confidence intervals and saves the pair data

        Returns:
            pairs dataset as a pandas dataframe
        """

        pair_df = pd.read_csv(_PAIRS_PATH, header=0, sep="\\s*,\\s*", usecols=["left", "right", "mean", "std_dev"])
        return pair_df
