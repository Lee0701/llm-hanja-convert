import mecab_ko as MeCab
from ollama_chat import prepare_context, gen_response

class LLMHanjaConverter:
    def __init__(self, model):
        self.model = model
        self.tagger = MeCab.Tagger()
        self.hanja_dict = load_hanja_dict()
        self.system_messages = prepare_context(self.model)
        self.messages = []

    def convert(self, text):
        tokens = self.tokenize(text)

        self.messages = self.messages[-1000:]

        result = []
        for start, surface, pos in tokens:
            if pos[0] == 'N':
                result.append(self.convert_token(text, start, surface))
            else:
                result.append(surface)

        return ''.join(result)

    def convert_token(self, text, start, surface):
        end = start + len(surface)
        marked_text = text[:start] + f'**{surface}**' + text[end:]
        candidates = self.hanja_dict.get(surface, []) + [surface]
        if len(candidates) == 1:
            return surface
        else:
            converted = self.select_candidate(marked_text, candidates)
            return converted

    def select_candidate(self, marked_text, candidates):
        text = marked_text + '\n' + ' '.join(candidates)

        result = ''
        while len(result) != len(candidates[-1]):
            self.messages.append({
                'role': 'user',
                'content': text,
            })
            messages = self.system_messages + self.messages
            result = gen_response(self.model, messages)
            if result is None:
                result = ''
            self.messages.append({
                'role': 'assistant',
                'content': result,
            })

        return result

    def tokenize(self, text):
        spaces = [i for i, c in enumerate(text) if c == ' ']
        tokens = self.tagger.parse(text).split('\n')
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
