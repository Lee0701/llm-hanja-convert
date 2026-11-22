
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
