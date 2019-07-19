import numpy as np
import torch

from autoencoder import network

_LAYERS_DICT = {
    0:  "shape",
    1:  "normal_bow",
    2:  "normal_cup",
    3:  "normal_pot",
    4:  "normal_book",
    5:  "normal_lamp",
    6:  "normal_button",
    7:  "effect_ball",
    8:  "effect_bomb",
    9:  "effect_rocket_up_down",
    10: "effect_rocket_left_right",
    11: "effect_plane",
    12: "background_oil",
    13: "background_rug",
    14: "special_cherry",
    15: "special_donut",
    16: "special_foam",
    17: "special_apple_box",
    18: "special_soap_foamer",
    19: "cover_box_1",
    20: "cover_box_2",
    21: "cover_box_3",
    22: "cover_ice_1",
    23: "cover_ice_2",
    24: "cover_chain_1",
    25: "cover_chain_2",
    26: "cover_jelly_1",
    27: "cover_jelly_2",
    28: "cover_jelly_3",
    29: "cover_jelly_4",
    30: "cover_jelly_5",
    31: "cover_cookie_1",
    32: "cover_cookie_2",
    33: "cover_cookie_3",
    34: "wall_info"
}
_ROWS = 12
_COLUMNS = 12
_CHANNELS = 35
_THRESHOLD = 0.3


class LevelGenerator(object):

    def __init__(self):
        """
        Class for generating levels back from encoded representation
        """

        # Initialize the AutoEncoder model
        self.model = network.AutoEncoderNetwork().double()
        self.model.load_state_dict(torch.load("./output/autoencoder/model.pth"))
        self.model.set_encoder_needed(False)
        self.model.set_decoder_needed(True)
        self.model.eval()

    def generate_level_data(self, vector):
        """
        Returns the level in original representation from the dense representation

        Args:
            vector: ndarray containing dense representation of level
        """

        level_tensor = self.model(torch.from_numpy(vector))
        level_data = self.convert_tensor_to_level(level_tensor)

        return level_data

    def convert_tensor_to_level(self, level_tensor):
        """
        Convert the Pytorch tensor into required level representation

        Args:
            level_tensor: torch tensor of level

        Returns:
            level in original representation from the dense representation
        """

        level_data = level_tensor.detach().numpy()[0]

        level_bin_data = np.zeros((_CHANNELS, _ROWS, _COLUMNS))
        for i in range(_CHANNELS):
            level_bin_data[i] = (level_data[i] >= _THRESHOLD).astype(int)

        level_py = {'level_index': 99999, 'move_count': 100, 'board_info': dict(), 'trans_info': dict()}
        apple_box = []

        for i in range(level_bin_data.shape[1]):
            for j in range(level_bin_data.shape[2]):
                element = dict()

                # Initialize empty grid cell
                if level_bin_data[0][i][j] == 1:
                    element['prev'] = (0, -1)
                    element['next'] = (0, 1)

                    if j == 0:
                        element['fall_point'] = (0, -1)

                # Check for base elements
                for k in range(1, 7):
                    if level_bin_data[k][i][j] == 1:
                        element['base'] = (k, 1)

                # Check for powerups
                for k in range(7, 12):
                    if level_bin_data[k][i][j] == 1:
                        element['base'] = (k + 3, 1)

                # Check for background
                for k in range(12, 14):
                    if level_bin_data[k][i][j] == 1:
                        element['bg_number'] = k + 28
                        element['background_number'] = k + 28

                # Check for specials
                for k in range(14, 17):
                    if level_bin_data[k][i][j] == 1:
                        element['base'] = (k + 36, 1)

                # Check for apple box
                if level_bin_data[17][i][j] == 1:
                    apple_box.append((i, j))

                # Check for foamer
                if level_bin_data[18][i][j] == 1:
                    element['base'] = (55, 1)

                # Check for box
                for k in range(19, 22):
                    if level_bin_data[k][i][j] == 1:
                        element['cover'] = (60, k - 18)

                # Check for ice
                for k in range(22, 24):
                    if level_bin_data[k][i][j] == 1:
                        element['cover'] = (61, k - 21)

                # Check for chain
                for k in range(24, 26):
                    if level_bin_data[k][i][j] == 1:
                        element['cover'] = (63, k - 23)

                # Check for jelly
                for k in range(26, 31):
                    if level_bin_data[k][i][j] == 1:
                        element['cover'] = (64, k - 25)

                # Check for cookie
                for k in range(31, 34):
                    if level_bin_data[k][i][j] == 1:
                        element['base'] = (66, k - 30)

                # Add data to board
                if element:
                    level_py['board_info'][(i, j)] = element

        # Add apple box data
        if len(apple_box) > 0:
            level_py['apple_box_info'] = list()
            level_py['apple_box_info'] = apple_box

        # Check for walls
        walls = self.get_walls_data(level_bin_data)
        if len(walls) > 0:
            level_py['wall_info'] = list()
            level_py['wall_info'] = walls

        level_py['trans_info'][(0, 0)] = {41: 99}

        level_str = "data = " + str(level_py)

        return level_str

    def get_walls_data(self, level_bin_data):
        """
        Get the walls data for the provided level

        Args:
            level_bin_data: ndarray containing binary encoding of level data

        Returns:
            list containing walls information in level
        """

        walls = []

        for i in range(_ROWS - 1):
            for j in range(_COLUMNS - 1):
                if level_bin_data[34][i][j] == 1:
                    if level_bin_data[34][i + 1][j] == 1:
                        walls.append([(i, j), (i + 1, j)])

                    if level_bin_data[34][i][j + 1] == 1:
                        walls.append([(i, j), (i, j + 1)])

        return walls
