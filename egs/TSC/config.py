import argparse

def load_args():
    parser = argparse.ArgumentParser()

    # training parameter
    parser.add_argument("--model_dir",type=str,default='./result')
    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--epochs",type=float,default=0.001)

    # Network parameter
    parser.add_argument("--num_hidden",type=int,default=32)
    parser.add_argument("--num_class",type=int,default=4)
    

    args = parser.parse_args()
    return args
