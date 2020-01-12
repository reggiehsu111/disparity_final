import argparse
from model import Model

parser = argparse.ArgumentParser(description='Train the model of cost computation')
parser.add_argument('--data', default='training/', type=str, help='path to training data')
# parser.add_argument('--data', default='data/Synthetic/', type=str, help='path to training data')

def train():
    args = parser.parse_args()

    model = Model()
    model.train(args.data)


if __name__ == '__main__':
    train()
