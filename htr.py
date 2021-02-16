import argparse

from project.htr.models import predict

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('char_table', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--mode', type=str, default='Greedy',)

    args = parser.parse_args()

    model_path = args.model
    image_path = args.image
    char_table_path = args.char_table
    raw = args.raw

    res = predict(
        model_path=model_path,
        char_table=char_table_path,
        image=image_path,
        decode_mode=args.mode
    )

    for key in res:
        if type(key) == int:
            continue

        print('Recognized text ({}): "{}"'.format(key, res[key]))
