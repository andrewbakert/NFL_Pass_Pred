import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_result_plot(grid, param):
    results = pd.DataFrame(grid.cv_results_)
    split_num = results.columns[results.columns.str.contains('split.*test_score')].shape[0]
    split = results.set_index(f'param_{param}')[[
        f'split{i}_test_score' for i in range(split_num)]].stack().reset_index()
    split.columns = [param, 'split', 'score']
    split.replace({f'split{i}_test_score': f'Split {i+1}'
                   for i in range(split_num)}, inplace=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_style('whitegrid')
    sns.lineplot(x=param, y='score', hue='split', data=split, ax=ax)
    ax.legend(title=False)
    ax.set_title('Scoring performance by split')