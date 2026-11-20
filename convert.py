import argparse
import mecab_ko as MeCab
from ollama import chat

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    tagger = MeCab.Tagger()
    hanja_dict = load_hanja_dict()

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        lines = 0
        for line in fin:
            line = line[:-1]
            result = convert(tagger, args.model, hanja_dict, line)
            fout.write(result + '\n')
            fout.flush()
            lines += 1
            if lines % 100 == 0:
                print(f'=== Processed {lines} lines ===')
                print()

def load_hanja_dict():
    with open('4888.txt', 'r') as f:
        hanja_list = set(f.read())
    result = dict()
    with open('hanja.txt', 'r') as f:
        for line in f:
            if line[0] == '#':
                continue
            line = line[:-1]
            tokens = line.split(':')
            if len(tokens) != 3:
                continue
            hangul, hanja, extra = tokens
            if hangul not in result:
                result[hangul] = []
            if len(hangul) == 1 and hanja not in hanja_list:
                continue
            result[hangul].append(hanja)
    return result

def convert(tagger, model, dictionary, text):
    tokens = tokenize(tagger, text)
    messages = reset_context(model)
    result = []
    for start, surface, pos in tokens:
        if pos[0] == 'N':
            end = start + len(surface)
            marked_text = text[:start] + f'**{surface}**' + text[end:]
            candidates = dictionary.get(surface, []) + [surface]
            if len(candidates) == 1:
                result.append(surface)
            else:
                input_text = marked_text + '\n' + ' '.join(candidates)
                converted = convert_token(model, messages, input_text)
                result.append(converted)
        else:
            result.append(surface)
    return ''.join(result)

def convert_token(model, messages, text):
    messages.append({
        'role': 'user',
        'content': text,
    })
    result = gen_response(model, messages)
    return result

def tokenize(tagger, text):
    spaces = [i for i, c in enumerate(text) if c == ' ']
    tokens = tagger.parse(text).split('\n')
    result = []
    length = 0
    for token in tokens:
        if length in spaces:
            result.append([length, ' ', '_'])
            length += 1
        if '\t' not in token:
            continue
        surface, features = token.split('\t')
        features = features.split(',')
        pos = features[0]
        result.append([length, surface, pos])
        length += len(surface)
    return result

def reset_context(model):
    with open('system_prompt.md', 'r') as f:
        system_prompt = f.read()
    messages = []
    messages.append({
        'role': 'system',
        'content': system_prompt,
    })
    gen_response(model, messages)
    return messages

def gen_response(model, messages) -> str:
    stream = chat(
        model=model,
        messages=messages,
        stream=True,
    )
    content = ''
    for chunk in stream:
        if chunk.message.thinking is not None:
            print(chunk.message.thinking, end='', flush=True)
        if chunk.message.content is not None:
            content += chunk.message.content
    print('\n')
    messages.append({
        'role': 'assistant',
        'content': content,
    })
    return content

if __name__ == '__main__':
    main()
