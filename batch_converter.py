import mecab_ko as MeCab
from hanja_dict import load_hanja_dict
from ollama import chat
import json

class BatchLLMHanjaConverter:
    def __init__(self, model):
        self.model = model
        self.tagger = MeCab.Tagger()
        self.hanja_dict = load_hanja_dict()
        with open('system_prompt_batch.md', 'r') as f:
            self.system_prompt = f.read()

    def convert(self, text):
        request, decoding = self.generate_request(text)
        decoding_segments = decoding['segments']

        result = None
        while result is None:
            response = self.gen_response(request)
            if response is None:
                continue
            if len(response) != len(decoding_segments):
                continue
            result_text = text
            for segment, selected_id in zip(decoding_segments.values(), response.values()):
                candidates = segment['candidates']
                if selected_id < 0 or selected_id >= len(candidates):
                    break
                start = segment['start']
                end = segment['end']
                selected_candidate = candidates[selected_id]
                result_text = result_text[:start] + selected_candidate + result_text[end:]
            else:
                result = result_text

        return result

    def gen_response(self, request):
        messages = []
        messages.append({
            'role': 'system',
            'content': self.system_prompt,
        })
        response = self.ollama_chat(messages)
        messages.append({
            'role': 'assistant',
            'content': response,
        })
        messages.append({
            'role': 'user',
            'content': json.dumps(request, indent=2, ensure_ascii=False),
        })
        response = self.ollama_chat(messages)
        try:
            response = json.loads(response)
        except json.JSONDecodeError:
            return None
        return response

    def ollama_chat(self, messages) -> str:
        try:
            stream = chat(
                model=self.model,
                messages=messages,
                stream=True,
                format='json',
            )
            content = ''
            for chunk in stream:
                if chunk.message.thinking is not None:
                    print(chunk.message.thinking, end='', flush=True)
                if chunk.message.content is not None:
                    content += chunk.message.content
            print('\n')
        except KeyboardInterrupt:
            raise
        except:
            return None
        return content

    def generate_request(self, text):
        tokens = self.tokenize(text)
        marked_text = []
        request_segments = {}
        decoding_segments = {}

        for start, surface, pos in tokens:
            end = start + len(surface)
            if pos[0] == 'N':
                segment_id = len(request_segments)
                candidates = self.hanja_dict.get(surface, []) + [surface]

                if len(candidates) == 1:
                    marked_text.append(surface)
                    continue

                marked_text.append(f'**[{segment_id}:{surface}]**')
                request_segments[segment_id] = [self.format_candidate(i, c) for i, c in enumerate(candidates)]
                decoding_segments[segment_id] = {
                    'start': start,
                    'end': end,
                    'surface': surface,
                    'candidates': candidates,
                }
            else:
                marked_text.append(surface)

        request_info = {
            'text': ''.join(marked_text),
            'segments': request_segments,
        }
        decoding_info = {
            'text': text,
            'segments': decoding_segments,
        }

        return request_info, decoding_info

    def format_candidate(self, index, candidate):
        return {
            'candidate_id': index,
            'candidate_text': candidate,
        }

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
