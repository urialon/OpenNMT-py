import seaborn as sns
from argparse import ArgumentParser
import pandas as pd
import matplotlib


def load_training_log(filename):
    steps = []
    training_accuracies = []
    validation_accuracies = []
    last_training_step = None
    last_training_acc = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.rstrip()
            parts = line.split(' ')

            if 'Validation' in parts:
                if 'accuracy:' in parts:
                    index_of_accuracy = parts.index('accuracy:')
                    val_acc = float(parts[index_of_accuracy + 1])
                    steps.append(last_training_step)
                    training_accuracies.append(last_training_acc)
                    validation_accuracies.append(val_acc)
                continue
            if 'Step' in parts:
                index_of_step = parts.index('Step')
                last_training_step = int(parts[index_of_step + 1].split('/')[0])
                index_of_acc = parts.index('acc:')
                last_training_acc = float(parts[index_of_acc + 2].rstrip(';'))
            
    return steps, training_accuracies, validation_accuracies


def load_eval_log(filename):
    steps = []
    top1_accuracies = []
    top5_accuracies = []
    last_top5 = None
    is_beam_acc = True
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            parts = line.split(' ')

            if 'Beam accuracy @k:' in line:
                is_beam_acc = True
                continue
            if 'Tree-only beam accuracy @k:' in line:
                is_beam_acc = False
                continue

            if len(parts) > 0 and parts[0] == 'Final:':
                step = int(parts[1].split('/')[-1].removeprefix('model_step_').removesuffix('.pt.txt'))
                top1_acc = float(parts[2]) * 100
                steps.append(step)
                top1_accuracies.append(top1_acc)
                top5_accuracies.append(last_top5)
                continue
            if 'accuracy@1:' in parts and not is_beam_acc:
                index_of_acc = parts.index('accuracy@1:') + 1
                last_top5 = float(parts[index_of_acc]) * 100
                
            
    return steps, top1_accuracies, top5_accuracies

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--baseline_log", required=True)
    parser.add_argument("--small_log", required=True)
    parser.add_argument("--small_baseline_log", required=True)
    # parser.add_argument("--train_log", required=True)
    # parser.add_argument("--baseline_train_log", required=True)
    args = parser.parse_args()

    steps, top1_accs, _ = load_eval_log(args.log)
    small_steps, small_top1_accs, _ = load_eval_log(args.small_log)
    baseline_steps, baseline_top1_accs, _ = load_eval_log(args.baseline_log)
    small_nofeat_steps, small_nofeat_top1_accs, _ = load_eval_log(args.small_baseline_log)
    # train_steps, training_accuracies, _ = load_training_log(args.train_log)
    # baseline_train_steps, baseline_training_accuracies, _ = load_training_log(args.baseline_train_log)
    
    
    main_df = pd.DataFrame({'steps': steps + small_steps + baseline_steps + small_nofeat_steps, 
                            'acc': top1_accs + small_top1_accs + baseline_top1_accs + small_nofeat_top1_accs, 
                            'type': ['top1_acc'] * len(top1_accs) 
                                    + ['top1_acc'] * len(small_top1_accs)
                                    + ['top1_acc'] * len(baseline_top1_accs)
                                    + ['top1_acc'] * len(small_nofeat_top1_accs),
                            'model': ['syntactic'] * len(steps)
                                    + ['syntactic256'] * len(small_steps)
                                    + ['baseline'] * len(baseline_steps)
                                    + ['baseline256'] * len(small_nofeat_steps)
                                    })
    # main_df = pd.DataFrame({'steps': steps + baseline_steps, 
    #                         'acc': top1_accs + baseline_top1_accs,
    #                         'model': ['syntactic'] * len(steps)
    #                                 + ['baseline'] * len(baseline_steps)
    #                                 })


    # ax = sns.scatterplot(data=main_df, x='steps', y='acc', style='type', hue='model', markers=True, s=5) # scatter_kws={'s':5}, line_kws={"color": "red"})
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
    # ax.set(ylim=(0, None))
    ax = sns.lineplot(data=main_df, x='steps', y='acc', style='type', hue='model', markers=True)
    xlabels = [f'{x//1000:.0f}K' for x in ax.get_xticks()]
    ax.set_xticklabels(xlabels)
    matplotlib.pyplot.show()

    # main_df = pd.DataFrame({'steps': train_steps + baseline_train_steps, 
    #                         'acc': training_accuracies + baseline_training_accuracies, 
    #                         'type': ['train_acc'] * len(train_steps)
    #                                 + ['train_acc'] * len(baseline_train_steps),
    #                         'model': ['syntactic'] * len(train_steps)
    #                                 + ['baseline'] * len(baseline_train_steps)})
    # ax = sns.lineplot(data=main_df, x='steps', y='acc', style='type', hue='model', markers=True)
    # xlabels = [f'{x//1000:.0f}K' for x in ax.get_xticks()]
    # ax.set_xticklabels(xlabels)
    # matplotlib.pyplot.show()