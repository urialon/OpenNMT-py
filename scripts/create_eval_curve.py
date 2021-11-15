import seaborn as sns
from argparse import ArgumentParser
import pandas as pd
import matplotlib

max_ppl = 1000

def load_training_log(filename):
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
                top1_acc = float(parts[2])
                steps.append(step)
                top1_accuracies.append(top1_acc)
                top5_accuracies.append(last_top5)
                continue
            if 'accuracy@1:' in parts and not is_beam_acc:
                index_of_acc = parts.index('accuracy@1:') + 1
                last_top5 = float(parts[index_of_acc])
                
            
    return steps, top1_accuracies, top5_accuracies

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--baseline_log", required=True)
    args = parser.parse_args()

    steps, top1_accs, top5_accs = load_training_log(args.log)
    baseline_steps, baseline_top1_accs, baseline_top5_accs = load_training_log(args.baseline_log)
    
    # main_df = pd.DataFrame({'steps': steps + steps + baseline_steps + baseline_steps, 
    #                         'acc': top1_accs + top5_accs + baseline_top1_accs + baseline_top5_accs, 
    #                         'type': ['top1_acc'] * len(top1_accs) 
    #                                 + ['top5_acc'] * len(top5_accs)
    #                                 + ['top1_acc'] * len(baseline_top1_accs)
    #                                 + ['top5_acc'] * len(baseline_top5_accs),
    #                         'model': ['syntactic'] * len(steps) * 2
    #                                 + ['baseline'] * len(baseline_steps) * 2})
    main_df = pd.DataFrame({'steps': steps + baseline_steps, 
                            'acc': top1_accs + baseline_top1_accs, 
                            'type': ['top1_acc'] * len(top1_accs) 
                                    + ['top1_acc'] * len(baseline_top1_accs),
                            'model': ['syntactic'] * len(steps) * 1
                                    + ['baseline'] * len(baseline_steps) * 1})

    # ax = sns.scatterplot(data=main_df, x='steps', y='acc', style='type', hue='model', markers=True, s=5) # scatter_kws={'s':5}, line_kws={"color": "red"})
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
    # ax.set(ylim=(0, None))
    matplotlib.pyplot.show()
    ax = sns.lineplot(data=main_df, x='steps', y='acc', style='type', hue='model', markers=True)
    xlabels = [f'{x//1000:.0f}K' for x in ax.get_xticks()]
    ax.set_xticklabels(xlabels)
    matplotlib.pyplot.show()
