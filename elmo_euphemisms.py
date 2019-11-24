from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder('/Users/kseniakostomarova/PycharmProjects/euphemisms/195/options.json', '/Users/kseniakostomarova/PycharmProjects/euphemisms/195/model.hdf5')
tokens = ''.split()
vectors = elmo.embed_sentence(tokens)

assert(len(vectors) == 3) # one for each layer in the ELMo output
assert(len(vectors[0]) == len(tokens)) # the vector elements correspond with the input tokens

import scipy
tokens2 = ''.split()
vectors2 = elmo.embed_sentence(tokens2)
print(scipy.spatial.distance.cosine(vectors[2][3], vectors2[2][3]))