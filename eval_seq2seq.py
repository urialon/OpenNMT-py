import numpy as np
from argparse import ArgumentParser
import re

def binary_to_string(binary_string):
    return binary_string.decode("utf-8")

def binary_to_string_list(binary_string_list):
    return [binary_to_string(w) for w in binary_string_list]

def load_lines(filename):
    with open(filename, 'rb') as file:
        lines = [line.rstrip() for line in file.readlines()]
        return binary_to_string_list(lines)

remove_names_regex = re.compile('[a-z]+')
remove_numbers_regex = re.compile('[0-9]+')
subtokens_regex = re.compile('\([^()]+\)')

def get_tree(seq):
    seq = ''.join(seq.split(' '))
    tree = re.sub(subtokens_regex, '()', seq)
    return tree

keywords = {'super', 'this', 'null', 'void', 'instanceof'}

def remove_names(s):
    joined_subtokens = s.replace(' _ ', '')
    remove_unks = joined_subtokens.replace('<unk>', 'w')
    for w in keywords:
        remove_unks = remove_unks.replace(w, w.upper())
    removed_numbers = remove_numbers_regex.sub('NUM', remove_unks)
    tree = remove_names_regex.sub('w', removed_numbers)
    return tree

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--expected", dest="expected", required=True)
    parser.add_argument("--actual", dest="actual", required=True)
    args = parser.parse_args()

    expected_lines = load_lines(args.expected)
    actual_lines = load_lines(args.actual)
    beam_size = len(actual_lines) // len(expected_lines)

    batched_hyps = [actual_lines[i*beam_size:(i+1)*beam_size] for i in range(len(expected_lines))]
    results = np.zeros(beam_size)
    tree_results = np.zeros(beam_size)
    count = 0
    most_common_correct_trees = {}
    for expected, hyps in zip(expected_lines, batched_hyps):
        count += 1
        if expected in hyps:
            index_of_correct = hyps.index(expected)
            update = np.concatenate(
                [np.zeros(index_of_correct, dtype=np.int32), np.ones(beam_size - index_of_correct, dtype=np.int32)])
            results += update

        tree_expected = remove_names(expected)
        tree_hyps = [remove_names(h) for h in hyps]
        if tree_expected in tree_hyps:
            index_of_correct = tree_hyps.index(tree_expected)
            update = np.concatenate(
                [np.zeros(index_of_correct, dtype=np.int32), np.ones(beam_size - index_of_correct, dtype=np.int32)])
            tree_results += update
            if tree_expected in most_common_correct_trees:
                most_common_correct_trees[tree_expected] += 1
            else:
                most_common_correct_trees[tree_expected] = 1

    accuracies = results / count
    tree_accuracies = tree_results / count

    print('Count: {}'.format(count))
    print('Beam size: {}'.format(beam_size))
    print('Beam accuracy @k:')
    for i, acc in enumerate(accuracies):
        print('\taccuracy@{}: {:.5f}'.format(i + 1, acc))


    print('Tree-only beam accuracy @k:')
    for i, acc in enumerate(tree_accuracies):
        print('\taccuracy@{}: {:.5f}'.format(i + 1, acc))

    print()
    s = [(k, most_common_correct_trees[k]) for k in
         sorted(most_common_correct_trees, key=most_common_correct_trees.get, reverse=True)]

    #for k,v in s:
    #    print(v, ': ', k)
