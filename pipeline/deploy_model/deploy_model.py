import argparse

def deploy_model(data_path):
    print(f'deploying model {data_path}...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    deploy_model(args.model)
