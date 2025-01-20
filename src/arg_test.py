import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-exp-slug', type=str, default='exp_', help='The slug of the experiment')
    parser.add_argument('--save-full-results', default=False, action='store_true')
    parser.add_argument('-d', type=str, action='append')
    args = parser.parse_args()
    print(args)


if __name__ == "__main__":
    main()
