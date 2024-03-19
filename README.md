# Deep Generative Model for Group Chat Simulation
This project explores the capability of generative models to simulate chat conversations. To understand and simulate chat conversations, there are many intricacies that need to be learned by the models, including the dynamics of group conversations and individual communication styles. In this project, we explore different ways to model a specific person's texting style by training a model from scratch and fine-tuning a pretrained model.

### To run the model
* To run LSTM-based models:
    - Navigate to the folder `LSTM/v1/` for our baseline model, a simple multilayered LSTM model, then run `python main.py --config config.json`.
    - Navigate to the folder `LSTM/seq2seq/` for the encoder-decoder LSTM model, then run `python main.py --config config_seq2seq.json`.
* Data can be found in the 'data' directory.
* The specifications of each experiment can be found in the config files. These files need to be in the 'configs' directory

### Team name
Unsupervised Learners

### Team members
Ester Tsai, Xian Ying Kong, Jonathan Cheung, Jeremy Tow, Samuel Chu
