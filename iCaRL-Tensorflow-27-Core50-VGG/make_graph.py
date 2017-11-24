import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def printInPlot(dataFrame, color, label=""):
    df1 = dataFrame
    df1 = df1.replace({'%': ''}, regex=True)
    # print(df1.T)
    df1 = df1.astype(np.float32)
    # plot = df1.T.plot()
    plot = df1.T.plot(y='avg', c=color, label=label)
    df2 = df1.T
    df2['min'] = df2['avg'] - df2['dev.std']
    df2['max'] = df2['avg'] + df2['dev.std']
    df2 = df2.astype(np.float32)
    # print(np.isfinite(df2))
    #print(df2)
    # plot.fill_between(x='RunID', y1='min', y2='max', data=df2)
    plot.fill_between(range(9), df2['min'], df2['max'], alpha=0.3, color=color)
    # plot.fill_between(range(9), df2['min']+1, df2['max'], alpha=0.5, color='yellow')
    return plot
def printIcaRL():
    #df1 = pd.read_csv('ICARL_VGG.tsv', sep='\t', header=0, skip_footer=0, index_col=0, skiprows=5)
    df1 = pd.read_csv('NC_no4.txt', sep='\t', header=0, skip_footer=0, index_col=0, skiprows=5)
    naive_df = pd.read_csv('NC.tsv', sep='\t', header=0, skip_footer=67, index_col=0, skiprows=54)
    reinit_df = pd.read_csv('NC.tsv', sep='\t', header=0, skip_footer=49, index_col=0, skiprows=73)
    cumulative_df = pd.read_csv('NC.tsv', sep='\t', header=0, skip_footer=36, index_col=0, skiprows=90)
    print(cumulative_df)
    plot = printInPlot(df1, 'gray', "iCaRL")
    plot = printInPlot(naive_df, 'red', "Naive")
    plot = printInPlot(reinit_df, 'green', "Weight reinit")
    plot = printInPlot(cumulative_df, 'blue', "Cumulative")
    plt.legend(loc=2)
    #plot = printInPlot(df3, 'blue', plot)
    print(type(plot))
    return plot

plot = printIcaRL()
fig = plot.get_figure()
fig.savefig('output.png')
