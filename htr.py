import argparse

from project.htr.models import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('char_table', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--raw', type=bool, default=False)

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    char_table_path = args.char_table
    raw = args.raw

    res = predict(model_path=model_path, char_table=char_table_path, image=image_path)

    print('Recognized text: "{}"'.format(res))
