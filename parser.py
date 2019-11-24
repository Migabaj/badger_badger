def csv_to_json(csv_file_path):
    import json


    with open(csv_file_path, encoding='utf-8') as f:
        text = f.readlines()

    euphemism_examples = []
    for line in text:
        euphemism_example = {}
        line = line.split('\t')
        euphemism_example['example'] = line[0]
        euphemism_example['euphemism_index'] = int(line[1])
        euphemism_example['substitution'] = line[2]
        euphemism_examples.append(euphemism_example)

    with open('euphemism_examples.json', 'w', encoding='utf-8') as f:
        json.dump(euphemism_examples, f, ensure_ascii=False, indent=4)

    return euphemism_examples


def does_pos_coincide(lemma1, lemma2):
    pos1 = lemma1.split('_')
    pos2 = lemma2.split('_')

    return pos1 == pos2


def distance(euphemism_example, index, neutral, synonym):
    euphemism_example = euphemism_example.split()

    temp = euphemism_example

    temp[index] = neutral
    neutral_example = temp

    temp[index] = synonym
    synonym_example = temp

    from allennlp.commands.elmo import ElmoEmbedder
    elmo = ElmoEmbedder('options.json', 'model.hdf5')
    import scipy

    vectors = elmo.embed_sentence(neutral_example)

    assert (len(vectors) == 3)  # one for each layer in the ELMo output
    assert (len(vectors[0]) == len(neutral_example))  # the vector elements correspond with the input tokens

    vectors2 = elmo.embed_sentence(synonym_example)
    neutral_synonym_distance = scipy.spatial.distance.cosine(vectors[2][index], vectors2[2][index])

    vectors2 = elmo.embed_sentence(euphemism_example)
    euphemism_synonym_distance = scipy.spatial.distance.cosine(vectors[2][index], vectors2[2][index])

    return neutral_synonym_distance, euphemism_synonym_distance
