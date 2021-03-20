import pytest
from scripts.NN import *
from scripts.io import *

@pytest.fixture
def test_make_weights():
    nn = NeuralNetwork(setup=[[8,3,"sigmoid"],[3,8,"sigmoid"]])
    # check that it made the correct number of layers and 
    # nodes per layer
    assert len(nn.weights) == 2
    assert len(nn.weights[0]) == 3
    assert len(nn.weights[1]) == 8
    assert len(nn.weights[0][0]) == 9
    assert len(nn.weights[1][0]) == 4

def test_feedforward():
    nn = NeuralNetwork([[2,1, "sigmoid"], [1,2, "sigmoid"]])
    # a 2x1x2 network
    out = nn.feedforward([1,1])
    # check that it gives the correct output length and 
    # if the outputs list got updated
    assert len(out) == 2
    assert nn.outputs[1] == out
    
def test_backprop():
    nn = NeuralNetwork([[2,1, "sigmoid"], [1,2, "sigmoid"]])
    nn.feedforward([1,1])
    old_weights = nn.weights
    nn.backprop(true_values = [1,1], data = [[1,1],[1,1]])
    new_weights = nn.weights
    # check that the weights actually got updated
    assert old_weights != new_weights
    

def test_read_train_seqs():
    pos,neg = read_train_seqs(pos_file = "data/rap1-lieb-positives.txt", neg_file = "data/yeast-upstream-1k-negative.fa")
    print(len(pos),len(neg))
    assert len(neg) == 3164
    assert len(pos) == 137
    for i in range(len(pos)):
        assert len(pos[i]) == 17

def test_encode_seq():
    seq = 'ACTG'
    encoded = encode_seq(seq)
    assert len(encoded) == 16
    assert encoded == [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]