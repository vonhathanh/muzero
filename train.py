import tyro

from args import Args

if __name__ == '__main__':
    args = tyro.cli(Args)
    print(args)
