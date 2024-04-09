import os
import argparse
import os.path as p 
# from main import main

PATH = p.dirname(__file__)

dataset_list = {'classification': ['DD','F194','CLEF','VOC','ILSVRC65','Sun','Cifar100','Car196']}

parser = argparse.ArgumentParser(description='HMGD')

parser.add_argument('--type','-t',type=str, default='classification',
                    help = 'classification')
parser.add_argument('--dataset_id','-id',type=int, default=7)

# training args
parser.add_argument('--epochs', '-e', type=int, default=200,
                    help = 'number of epochs to train (default: 500)')
parser.add_argument('--learning_rate','-lr', type=float, default=0.001,
                    help = 'learning rate (default: 0.01)')
parser.add_argument('--keep_prob','-k', type=float, default=0.9,
                    help = 'keep ratio of the dropout settings (default: 0.9)')
parser.add_argument('--batch_size','-batch_size', type=int, default=64,
                    help = 'batch_size')
parser.add_argument('--relabel', '-re', type=float, default=0,
                    help = 'ratio of relabeling')

# model args
parser.add_argument('--n_hidden','-hidden', type=int, default=512,
                    help = 'number of the hidden nodes (default: 150)')
parser.add_argument('--dim_z','-dim_z', type=int, default=200,
                    help='dimension of the variable Z (default: 100)')
parser.add_argument('--alpha','-a', type=float, default=0.1,
                    help = 'balance parameter of the loss function (default=1.0)')

# other args
parser.add_argument('--gpu', '-gpu', type = int, default = 1,
                    help = 'device of gpu id (default: 0)')
parser.add_argument('--seed', '-seed', type = int, default = 0,
                    help = 'random seed (default: 0)')

args = vars(parser.parse_args())

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    assert args['type'] in ['recovery', 'classification']
    args['dataset'] = dataset_list[args['type']][args['dataset_id']]
    print('---------------------------------------------------------------------')
    from main_hmgd import main
    print(args)
    main(args)



