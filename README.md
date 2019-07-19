# Match-3 Level Similarity Analysis

A Python3 project for generating a similarity score for any 2 levels of a Match-3 game.
This project includes 2 approaches as possible solutions: one method uses unsupervised 
learning with the help of a Simple AutoEncoder which was developed using PyTorch, and 
the other method is a rule-based algorithm based on the aggregation of level data for 
vectorization.

## Dependencies

The dependencies of the project are available in `requirements.txt` in the root of 
the project. They can be installed with the following command:
```
pip install -r requirements.txt
```

## Running the project

Run the project using the following command:
```
python main.py [algorithm]
```

where `algorithm` can take up the following values:
1. Train the model
2. Generate dense representation and level data
3. Generate scores for all levels
4. Optimize the weights for the vectors
5. Generate the plots
6. Rule-based algorithm score generation

Options `1-5` are for the AutoEncoder and option `6` is for the Rule-based algorithm

#### Note:

The pre-trained model is not available in this repository, but the model can easily trained and saved.
Just execute `python main.py 1` after installing the dependencies and follow the instructions to 
save the model. The model is saved and accessed by default at the path `./output/autoencoder/model.pth` 

## Report

* The report for this project can be accessed [here](output/Report.pdf)

## Built With

* [PyTorch](https://pytorch.org/) - The ML framework used 

## Author

* **Salil Deshpande** - Master of Information Systems Management Student at Carnegie Mellon University, USA - [Salild1011](https://github.com/Salild1011)

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

This project was given the right direction and was developed under the guidance of Mr. Simon Cheng Liu and 
Dr. Guo Xianghao of Levelup AI, Beijing.
