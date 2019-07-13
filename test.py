from a3c import *

'''
@authors:
Nicklas Hansen

Run a test with:
python test.py FOLDER
where FOLDER is the name of an auto-generated folder in /results.
'''

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str)
    parser.add_argument('--print_args', type=bool, default=False)
    args = parser.parse_args()

    # Load args
    args2 = load_args(args.data)
    if args.print_args is True:
        print(args2)

    # Initialize environment
    gym.logger.set_level(40)
    _env = gym.make(args2.env)
    args2.conv = len(_env.observation_space.shape) == 3
    model = ConvAC(args2) if args2.conv == True else AC(args2)

    # Load model
    model = load_model(model, args.data)
    model.eval()
    print('Initializing testing...')

    # Run
    test(model, args2)
