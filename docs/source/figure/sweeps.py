import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects

SIZE = 16

plt.rc('font', size=SIZE)
plt.rc('axes', titlesize=SIZE)
plt.rc('axes', labelsize=SIZE)
plt.rc('xtick', labelsize=SIZE)
plt.rc('ytick', labelsize=SIZE)
plt.rc('legend', fontsize=SIZE)
plt.rc('figure', titlesize=SIZE)
plt.rcParams['legend.title_fontsize'] = SIZE

# Define the path effects for nodes and edges with adjusted linewidth for node edges
node_path_effects = [patheffects.withStroke(linewidth=2, foreground='black')]
edge_path_effects = [patheffects.withStroke(linewidth=4, foreground='black')]

# Define a function to draw a plot in xkcd style showing training loss against learning rate with transparent background, no grid lines, and no ticks or tick labels
def draw_xkcd_training_loss_plots():
    plt.xkcd()  # Enable xkcd style
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_alpha(0)  # Make the background transparent
    
    # Data for the left plot (varying width)
    learning_rates = np.linspace(0, 1, 400)  # Use a linear range for learning rates
    widths = [2**i for i in range(5, 11)]
    colors = plt.cm.viridis(np.linspace(0, 1, len(widths)))
    optima_list = []
    
    for width, color in zip(widths, colors):
        # Generate a quadratic U-shaped curve for each width
        training_loss = (learning_rates - 0.3)**2 + 2 / np.log2(width)  # Example quadratic curve
        
        # Increase the drift of the minimum left
        drift = np.log2(width)
        shifted_learning_rates = learning_rates - drift / 10

        min_idx = np.argmin(training_loss)
        optima_list.append((shifted_learning_rates[min_idx], training_loss[min_idx]))
        
        with plt.style.context({'path.effects': []}):
            axes[0].plot(shifted_learning_rates, training_loss, label=f'{width}', color=color)
    
    axes[0].set_xlabel('learning rate')
    axes[0].set_ylabel('training loss')
    axes[0].set_title('optimal learning rate drifts')
    
    # Create the legend for the left plot
    legend = axes[0].legend(title='width', loc='center left', bbox_to_anchor=(1, 0.5))
    legend.get_frame().set_alpha(0)  # Remove the legend background color
    
    axes[0].grid(False)  # Disable grid lines
    axes[0].set_xticks([])  # Remove x ticks
    axes[0].set_yticks([])  # Remove y ticks
    axes[0].set_xticklabels([])  # Remove x tick labels
    axes[0].set_yticklabels([])  # Remove y tick labels
    axes[0].patch.set_alpha(0)  # Remove the subplot background

    x0, y0 = optima_list[-1]
    x1, y1 = optima_list[0]
    axes[0].annotate('', xy=(x0, y0), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', lw=4, mutation_scale=20, zorder=20))
    
    # Data for the right plot (varying depth)
    depths = [2**i for i in range(5, 11)]
    colors = plt.cm.viridis(np.linspace(0, 1, len(depths)))
    optima_list = []
    
    for depth, color in zip(depths, colors):
        # Generate a quadratic U-shaped curve for each depth with larger loss for deeper networks
        training_loss = (learning_rates - 0.3)**2 + np.log2(depth) / 10  # Example quadratic curve with larger loss for deeper networks

        min_idx = np.argmin(training_loss)
        optima_list.append((learning_rates[min_idx], training_loss[min_idx]))
        
        with plt.style.context({'path.effects': []}):
            axes[1].plot(learning_rates, training_loss, label=f'{depth}', color=color)
    
    axes[1].set_xlabel('learning rate')
    axes[1].set_ylabel('training loss')
    axes[1].set_title('deeper performs worse')
    
    # Create the legend for the right plot
    legend = axes[1].legend(title='depth', loc='center left', bbox_to_anchor=(1, 0.5))
    legend.get_frame().set_alpha(0)  # Remove the legend background color
    
    axes[1].grid(False)  # Disable grid lines
    axes[1].set_xticks([])  # Remove x ticks
    axes[1].set_yticks([])  # Remove y ticks
    axes[1].set_xticklabels([])  # Remove x tick labels
    axes[1].set_yticklabels([])  # Remove y tick labels
    axes[1].patch.set_alpha(0)  # Remove the subplot background

    x0, y0 = optima_list[-1]
    x1, y1 = optima_list[0]
    axes[1].annotate('', xy=(x0, y0), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color='red', lw=4, mutation_scale=20, zorder=20))
    
    plt.subplots_adjust(left=0.04, bottom=None, right=0.88, top=None, wspace=0.8, hspace=None)
    plt.show()

# Draw the training loss plots in xkcd style with transparent background, no grid lines, no ticks, no tick labels, and legends
draw_xkcd_training_loss_plots()
