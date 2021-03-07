import argparse
from utils_ic import load_data, read_jason
from model_ic import NN_Classifier, validation, make_NN, make_NN_CNN, make_NN_resnet, save_checkpoint

parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_dir", help="save model")
parser.add_argument("--num_layers", type=int, default=1, help="set number of layers for custom CNN")
parser.add_argument("--training_pref", default="finetune_top", help="choose model training preference")
parser.add_argument("--plot_graph", type=bool, default=False, help="plot train and val loss-step graph")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

if args.arch == "resnet18":
    model = make_NN_resnet(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, \
                           device=args.gpu, model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)
elif args.arch == "custom":
    model = make_NN_CNN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                    model_name=args.arch, num_layers=args.num_layers, plot_graph=args.plot_graph, trainloader=trainloader, \
                    validloader=validloader, train_data=train_data)
else: # default: densenet169
    model = make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                    model_name=args.arch, training_pref=args.training_pref, plot_graph=args.plot_graph, trainloader=trainloader, \
                    validloader=validloader, train_data=train_data)
    
if args.save_dir:
    save_checkpoint(model, args.save_dir)