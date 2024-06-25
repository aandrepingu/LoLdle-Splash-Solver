import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CHAMPIONS = 167
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparameters
epochs = 5
batch_size = 4
lr = 0.001


def mapping(tokens):
    
    # Generates 2-way mapping of tokens (subgraphs) to numbers
    token_to_id = {}
    id_to_token = {}

    for i, token in enumerate(set(tokens)):
        token_to_id[token] = i
        id_to_token[i] = token
    return token_to_id, id_to_token

if __name__ == "__main__":

    with open('./champnames.txt', 'r') as file:
        champ_names = file.readlines()

    assert len(champ_names) == NUM_CHAMPIONS
    champ_to_id, id_to_champ = mapping(champ_names)
    del champ_names

    # load images ferom splash directory
