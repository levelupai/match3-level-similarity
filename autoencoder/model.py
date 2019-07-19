import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data

from sklearn.model_selection import train_test_split
from torch.optim import rmsprop
from autoencoder import network, dataset

# Config
_LEARNING_RATE = 0.0001
_BATCH_SIZE = 24
_NUM_WORKERS = 0
_EPOCHS = 10
_MODEL_NAME = "./output/autoencoder/model.pth"


class Encoder(object):

    def __init__(self):
        """
        Class for training and testing the Convolutional AutoEncoder model
        """

        # Initialize the device
        self.device = self.get_device()

        # Initialize the model
        self.model = network.AutoEncoderNetwork().double()
        self.model.to(self.device)

        # Load the dataset and split it for training and testing
        full_dataset = dataset.LevelsDataset().get_data()
        self.train_dataset, self.test_dataset = train_test_split(full_dataset, test_size=0.2, random_state=42)

    def train_model(self):
        """
        Train the AutoEncoder model
        """

        # Fetch the training data
        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS)

        # Keep track of progress
        counter = []
        loss_history = []
        iteration_number = 0

        # Set the loss and optimizer functions
        criterion = nn.MSELoss()
        optimizer = rmsprop.RMSprop(self.model.parameters(), lr=_LEARNING_RATE)

        for epoch in range(0, _EPOCHS):
            for i, data in enumerate(train_dataloader, 0):
                # Get data
                level_data = data
                level_data = level_data.to(self.device)

                # Clear gradients
                optimizer.zero_grad()

                # Perform a forward pass
                outputs = self.model(level_data)

                # Calculate the loss
                loss = criterion(outputs, level_data)

                # Perform a backward pass
                loss.backward()

                # Parameter update - optimization step
                optimizer.step()

                if i % 10 == 0:
                    print("Epoch number {}: Current loss {}\n".format(epoch, loss.item()))
                    iteration_number += 10
                    counter.append(iteration_number)
                    loss_history.append(loss.item())

        # Plot the loss history
        self.show_plot(counter, loss_history)

    def test_model(self):
        """
        Test the AutoEncoder model
        """

        print("-" * 40)

        # Set the model to testing mode
        self.model.eval()

        # Fetch the testing data
        test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=_BATCH_SIZE, num_workers=_NUM_WORKERS)

        # Test all the data and its output
        for level_data in test_dataloader:
            level_data = level_data.to(self.device)
            outputs = self.model(level_data)

            loss = nn.MSELoss()
            print("\nTesting losses:")
            # Find the losses
            for x, y in zip(outputs, level_data):
                print(loss(x, y).item())

        # Check if user wants to save the model
        self.save_model()

    def show_plot(self, X, y):
        """
        Plot the provided series

        Args:
            X: The x-axis values
            y: The y-axis values
        """

        plt.plot(X, y)
        plt.show()

    def save_model(self):
        """
        Depending on user choice, saves the AutoEncoder model
        """

        print("-" * 40)
        print()

        choice = input("Do you want to save the model?(Y/n): ")

        if choice.lower() == "y":
            torch.save(self.model.state_dict(), _MODEL_NAME)
            print("Model has been saved as auto_encoder_model.pth")

        else:
            print("Model has not been saved")

    def get_device(self):
        """
        Returns the available device
        """

        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
