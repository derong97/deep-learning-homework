import argparse
from model_ic import load_model, test_model
from utils_ic import load_data

parser = argparse.ArgumentParser(description="Evaluate testing set")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("checkpoint", help="set path to checkpoint")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")

args = parser.parse_args()

model_cp = load_model(args.checkpoint)
trainloader, testloader, validloader, train_data = load_data(args.data_dir)
test_model(model_cp, testloader, device=args.gpu)