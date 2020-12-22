import argparse

def load_args():
    parser = argparse.ArgumentParser()

    # training parameter
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=float,default=100)
    parser.add_argument("--lr",type=float,default=0.001)
    parser.add_argument("--loss_type",type=str,default='CROSS')

    # Network parameter
    parser.add_argument("--hidden_num",type=int,default=32)
    parser.add_argument("--num_class",type=int,default=4)
    parser.add_argument("--filter_num",type=int,default=75)
    

    args = parser.parse_args()
    return args
