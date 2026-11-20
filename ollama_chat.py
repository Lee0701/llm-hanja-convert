from ollama import chat

def prepare_context(model):
    with open('system_prompt.md', 'r') as f:
        system_prompt = f.read()
    messages = []
    messages.append({
        'role': 'system',
        'content': system_prompt,
    })
    content = gen_response(model, messages)
    messages.append({
        'role': 'assistant',
        'content': content,
    })
    return messages

def gen_response(model, messages) -> str:
    try:
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
    except:
        return None
    return content
