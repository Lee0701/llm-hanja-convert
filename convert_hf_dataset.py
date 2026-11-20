import argparse
from converter import LLMHanjaConverter
from datasets import load_dataset
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--fields', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    converter = LLMHanjaConverter(args.model)

    ds = load_dataset(args.dataset, streaming=True)

    with open(args.output, 'w') as f:
        for example in ds['train']:
            try:
                for field in args.fields.split(','):
                    example[field] = convert_multiline(converter, example[field])
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
                f.flush()
            except KeyboardInterrupt:
                break

def convert_multiline(converter, text):
        lines = text.split('\n')
        lines = [converter.convert(line) for line in lines]
        return '\n'.join(lines)

if __name__ == '__main__':
    main()
