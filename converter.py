def converter(token):
    from pymorphy2 import MorphAnalyzer

    morph = MorphAnalyzer

    ana = morph.parse(token)
    pos = morph.[]