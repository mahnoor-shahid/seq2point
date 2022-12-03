
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

def get_maximum_step(traces: list):
    highest_val = 0
    for trace in traces:
        if highest_val == 0:
            highest_val = len(trace)
        else:
            if highest_val < len(trace):
                highest_val = len(trace)
    return highest_val
            

def plot_traces(traces: list, labels: list, axis_labels: list, colors = None,  title = None): 
    """
    
    """
    try:
        print(f"\nFollowings are the {PLOT_CONFIG['DESCRIPTION']} of your project..")
        pprint(PLOT_CONFIG)

        plt.style.use(PLOT_CONFIG['STYLE'])
        plt.rcParams['text.usetex'] = PLOT_CONFIG['LATEX']
        plt.rcParams['legend.handlelength'] = PLOT_CONFIG['LEGEND_HANDLE']
        plt.rcParams['font.size'] = PLOT_CONFIG['FONT_SIZE']
        plt.rcParams['legend.fontsize'] = PLOT_CONFIG['LEGEND_FONT_SIZE']
        plt.rcParams['xtick.direction'] = PLOT_CONFIG['TICK_DIRECTION']
        plt.rcParams['ytick.direction'] = PLOT_CONFIG['TICK_DIRECTION']
        plt.rcParams['xtick.major.size'] = PLOT_CONFIG['MAJOR_TICKS']
        plt.rcParams['xtick.minor.size'] = PLOT_CONFIG['MINOR_TICKS']
        plt.rcParams['ytick.major.size'] = PLOT_CONFIG['MAJOR_TICKS']
        plt.rcParams['ytick.minor.size'] = PLOT_CONFIG['MINOR_TICKS'] 
        
        fig, ax = plt.subplots(figsize=(PLOT_CONFIG['FIG_XSIZE'], PLOT_CONFIG['FIG_YSIZE']))
        x = np.arange(1, get_maximum_step(traces)+1, 1)
        for indx, trace in enumerate(traces):
            if colors is not None:
                ax.plot(x, trace, marker=PLOT_CONFIG['MARKER'], color=colors[indx], label=labels[indx])
            else:
                ax.plot(x, trace, marker=PLOT_CONFIG['MARKER'], color=PLOT_CONFIG['COLORS'][indx], label=labels[indx])
        ax.set_title(title, alpha=PLOT_CONFIG['OPACITY'])
        ax.set_xlabel(axis_labels[0], alpha=PLOT_CONFIG['OPACITY'])
        ax.set_ylabel(axis_labels[1], alpha=PLOT_CONFIG['OPACITY'])
        plt.yticks(alpha=PLOT_CONFIG['OPACITY'])
        plt.xticks(alpha=PLOT_CONFIG['OPACITY'])
        ax.set_xticks(x)
        ax.grid(True)
        plt.legend(loc=PLOT_CONFIG['LEGEND_LOCATION'])
        fig.tight_layout()
        plot_path = os.path.join(TRAINING_CONFIG['EXPERIMENT_PATH'], 'plots')
        plt.savefig(f'{os.path.join(plot_path, title)}.png')
            
    except Exception as e:
        print("Error occured in plot_traces method due to ", e)