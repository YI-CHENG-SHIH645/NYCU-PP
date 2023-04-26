import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('elapsed.csv', header=None)
    df.columns = ['experiment', 'time(sec)']
    df = df.groupby('experiment')['time(sec)'].mean().round(2)
    exps = ['pi_block_linear', 'block_tree', 'nonblock_linear', 'gather', 'reduce']
    for exp in exps:
        select = np.array([idx for idx in df.index if exp in idx])
        select = select[np.argsort([int(re.findall(r'\d+', s)[0]) for s in select])]
        os.makedirs('fig', exist_ok=True)
        plt.bar(df[select].index, df[select])
        plt.xticks(select, [2, 4, 8, 12, 16])
        plt.xlabel('num proc.', fontsize="10")
        plt.ylabel('time(sec)', fontsize="10")
        plt.title(exp, fontsize="18")
        for index, value in enumerate(df[select].values):
            plt.text(index-.15, value+.1, str(value), fontdict=dict(fontsize=15))
        plt.savefig(os.path.join('fig', exp+'.png'))
        plt.clf()
