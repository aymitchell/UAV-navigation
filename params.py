import argparse

'''
Get tunable hyperparameters using argparse.
Ex: --batch_size 32
'''

def get_params():
    parser = argparse.ArgumentParser(description='SocialLSTM Network')

    parser.add_argument('--batch_size', type=int, default=32,
                    help='batch size for model training')

    parser.add_argument('--startepoch', type=int, default=0,
                help='epoch to begin training')

    parser.add_argument('--nepochs', type=int, default=1,
                help='number of epochs to train for')

    parser.add_argument('--load_model', type=str, default=None,
                help='which model file to load')

    parser.add_argument('--load_optimizer', type=str, default=None,
                help='which optimizer file to load')

    params = parser.parse_args()

    return params