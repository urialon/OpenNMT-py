import numpy as np
from argparse import ArgumentParser
import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

target_num_bins = 40
target_min = 0
target_max = 40

src_num_bins = 25
src_min = 0
src_max = 2000

def get_bin(value, num_bins, min, max):
    bin = int(value // ((max - min) / num_bins))
    if bin >= num_bins:
        bin = num_bins - 1
    return bin

def binary_to_string(binary_string):
    return binary_string.decode("utf-8")

def binary_to_string_list(binary_string_list):
    return [binary_to_string(w) for w in binary_string_list]

def load_lines(filename):
    with open(filename, 'rb') as file:
        lines = [line.rstrip() for line in file.readlines()]
        return binary_to_string_list(lines)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--expected", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--actual_model", required=True)
    parser.add_argument("--actual_baseline", required=True)
    args = parser.parse_args()

    expected_lines = load_lines(args.expected)
    src_lines = load_lines(args.src)
    actual_model_lines = load_lines(args.actual_model)
    actual_baseline_lines = load_lines(args.actual_baseline)
    beam_size = len(actual_model_lines) // len(expected_lines)

    batched_model_hyps = [actual_model_lines[i*beam_size:(i+1)*beam_size] for i in range(len(expected_lines))]
    batched_baseline_hyps = [actual_baseline_lines[i*beam_size:(i+1)*beam_size] for i in range(len(expected_lines))]

    model_src_correct = np.zeros(src_num_bins)
    model_src_count = np.zeros(src_num_bins)
    model_target_correct = np.zeros(target_num_bins)
    model_target_count = np.zeros(target_num_bins)

    baseline_src_correct = np.zeros(src_num_bins)
    baseline_src_count = np.zeros(src_num_bins)
    baseline_target_correct = np.zeros(target_num_bins)
    baseline_target_count = np.zeros(target_num_bins)

    for src_line, expected, hyps in zip(src_lines, expected_lines, batched_model_hyps):
        src_bin = get_bin(len(src_line.split()), src_num_bins, src_min, src_max)
        target_bin = get_bin(len(hyps[0].split()), target_num_bins, target_min, target_max)
        if expected == hyps[0]:
            model_src_correct[src_bin] += 1
            model_target_correct[target_bin] += 1
        model_src_count[src_bin] += 1
        model_target_count[target_bin] += 1                        

    for src_line, expected, hyps in zip(src_lines, expected_lines, batched_baseline_hyps):
        src_bin = get_bin(len(src_line.split()), src_num_bins, src_min, src_max)
        target_bin = get_bin(len(hyps[0].split()), target_num_bins, target_min, target_max)
        if expected == hyps[0]:
            baseline_src_correct[src_bin] += 1
            baseline_target_correct[target_bin] += 1
        baseline_src_count[src_bin] += 1
        baseline_target_count[target_bin] += 1

    src_df = pd.DataFrame({'src_size': list(range(src_num_bins)) + list(range(src_num_bins)), 
                            'acc': list(model_src_correct / model_src_count) + list(baseline_src_correct / baseline_src_count),
                            'model': ['syntactic256'] * src_num_bins 
                            + ['baseline'] * src_num_bins
                            })
    ax = sns.lineplot(data=src_df, x='src_size', y='acc', style='model', hue='model', markers=True)
    xlabels = [int(x * (src_max - src_min) / src_num_bins) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabels)
    matplotlib.pyplot.show()

    min_src_df = pd.DataFrame({'min_src_size': list(range(src_num_bins)) + list(range(src_num_bins)), 
                            'acc': list(model_src_correct[::-1].cumsum()[::-1] / model_src_count[::-1].cumsum()[::-1]) 
                            + list(baseline_src_correct[::-1].cumsum()[::-1] / baseline_src_count[::-1].cumsum()[::-1]),
                            'model': ['syntactic256'] * src_num_bins 
                            + ['baseline'] * src_num_bins
                            })
    ax = sns.lineplot(data=min_src_df, x='min_src_size', y='acc', style='model', hue='model', markers=True)
    xlabels = [int(x * (src_max - src_min) / src_num_bins) for x in ax.get_xticks()]
    ax.set_xticklabels(xlabels)
    matplotlib.pyplot.show()

    # for src_line, expected, hyps, baseline_hyps in zip(src_lines, expected_lines, batched_model_hyps, batched_baseline_hyps):
    #     src_bin = get_bin(len(src_line.split()), src_num_bins, src_min, src_max)
       
    #     if expected == hyps[0] and expected != baseline_hyps[0]:
    #         with open(f'logs/examples/bin{int(src_bin * (src_max - src_min) / src_num_bins)}.txt', 'a') as file:      
    #             file.write(f'Context: \n{src_line}\n')
    #             file.write(f'Ground truth: {expected.replace(" ", "")}\n')
    #             file.write(f'Syntactic prediction: {hyps[0].replace(" ", "")}\n')
    #             file.write(f'Baseline prediction: {baseline_hyps[0].replace(" ", "")}\n')
    #             file.write('\n')

    # tgt_df = pd.DataFrame({'tgt_size': list(range(target_num_bins)) + list(range(target_num_bins)), 
    #                         'acc': list(model_target_correct / model_target_count) + list(baseline_target_correct / baseline_target_count),
    #                         'model': ['syntactic'] * target_num_bins 
    #                         + ['baseline'] * target_num_bins
    #                         })
    # ax = sns.lineplot(data=tgt_df, x='tgt_size', y='acc', style='model', hue='model', markers=True)
    # xlabels = [int(x * (target_max - target_min) / target_num_bins) for x in ax.get_xticks()]
    # ax.set_xticklabels(xlabels)
    # matplotlib.pyplot.show()

    # for bin in range(target_num_bins):
    #     model_acc = model_target_correct[bin:].sum() / model_target_count[bin:].sum() * 100
    #     baseline_acc = baseline_target_correct[bin:].sum() / baseline_target_count[bin:].sum() * 100
    #     print(f'{int(bin * (target_max - target_min) / target_num_bins)}: Syntactic: {model_acc:.0f}, baseline: {baseline_acc:.0f}')

    # count_df = pd.DataFrame({'src_size': list(range(src_num_bins)), 
    #                         'count': list(model_src_count),
    #                         })
    # ax = sns.barplot(data=count_df, x='src_size', y='count')
    # xlabels = [int(x * (src_max - src_min) / src_num_bins) for x in ax.get_xticks()]
    # ax.set_xticklabels(xlabels)
    # matplotlib.pyplot.show()