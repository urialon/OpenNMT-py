import seaborn as sns
from argparse import ArgumentParser
import pandas as pd
import matplotlib

max_ppl = 1000

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

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--train_log", required=True)
    parser.add_argument("--baseline_train_log", required=True)
    args = parser.parse_args()

    steps, training_accuracies, validation_accuracies = load_training_log(args.train_log)
    baseline_steps, baseline_training_accuracies, baseline_validation_accuracies = load_training_log(args.baseline_train_log)
    
    main_df = pd.DataFrame({'steps': steps + steps + baseline_steps + baseline_steps, 
                            'acc': training_accuracies + validation_accuracies + baseline_training_accuracies + baseline_validation_accuracies, 
                            'type': ['train_acc'] * len(training_accuracies) 
                                    + ['val_acc'] * len(validation_accuracies)
                                    + ['train_acc'] * len(baseline_training_accuracies)
                                    + ['val_acc'] * len(baseline_validation_accuracies),
                            'model': ['syntactic'] * len(steps) * 2
                                    + ['baseline'] * len(baseline_steps) * 2})

    ax = sns.scatterplot(data=main_df, x='steps', y='acc', style='type', hue='model', markers=True, s=5) # scatter_kws={'s':5}, line_kws={"color": "red"})
    # ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(50))
    # ax.set(ylim=(0, None))
    matplotlib.pyplot.show()
