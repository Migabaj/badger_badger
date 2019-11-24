import json

def write_examples():
    euphemisms = [{'example': 'Мне нужно в домик', 'euphemism_index': 3}]
    with open('euphemism_examples.json', 'w', encoding='utf-8') as f:
        json.dump(euphemisms, f, ensure_ascii=False, indent=4)

def closest_words(sentence, n=10):
    import gensim

