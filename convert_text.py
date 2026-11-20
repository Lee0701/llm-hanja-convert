import argparse
from converter import LLMHanjaConverter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    converter = LLMHanjaConverter(args.model)

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        lines = 0
        for line in fin:
            line = line[:-1]
            result = converter.convert(line)
            fout.write(result + '\n')
            fout.flush()
            lines += 1
            if lines % 100 == 0:
                print(f'=== Processed {lines} lines ===')
                print()

if __name__ == '__main__':
    main()
