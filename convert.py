import argparse
from ollama import chat
from ollama import ChatResponse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    messages = reset_context(args.model)

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        lines = 0
        for line in fin:
            line = line[:-1]
            messages.append({
                'role': 'user',
                'content': line,
            })
            response = gen_response(args.model, messages)
            fout.write(response + '\n')
            fout.flush()
            lines += 1
            if lines % 100 == 0:
                print(f'Processed {lines} lines')
                messages = reset_context(args.model)

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
