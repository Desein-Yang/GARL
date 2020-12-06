import argparse
def parse_train():
    parser = argparse.ArgumentParser(description="Tain WGAN model")
    parser.add_argument('--try1',dest='d1',default=1, type = int, help = "this is try1")
    parser.add_argument('--try2',dest='d2',default=2, type = int, help = "this is try2")
    args = parser.parse_args()
    return args

def main():
    args = parse_train()
    wgan = vars(args)
    print(type(wgan))
    print(wgan)

if __name__ == "__main__":
    main()
