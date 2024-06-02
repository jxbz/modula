import matplotlib.pyplot as plt
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

# Define a function to draw a neural network
def draw_xkcd_neural_net(layers, spacing=3):
    plt.xkcd()  # Enable xkcd style
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_alpha(0)  # Make the background transparent
    
    # Draw nodes layer by layer
    node_positions = []
    current_layer_x = 0
    for layer_index, num_nodes in enumerate(layers):
        layer_positions = []
        for node_index in range(num_nodes):
            x, y = current_layer_x, num_nodes/2-node_index
            layer_positions.append((x, y))
            # Draw nodes as smaller circles on top of lines with thinner black borders and darker color
            circle = plt.Circle((x, y), 0.25, edgecolor='black', facecolor='steelblue', lw=1, zorder=10)
            circle.set_path_effects(node_path_effects)
            ax.add_patch(circle)
        node_positions.append(layer_positions)
        current_layer_x += spacing

    # Draw edges between layers with increased thickness and darker color
    for layer_index in range(len(layers) - 1):
        for source_index, (source_x, source_y) in enumerate(node_positions[layer_index]):
            for target_index, (target_x, target_y) in enumerate(node_positions[layer_index + 1]):
                line, = ax.plot([source_x, target_x], [source_y, target_y], color='indianred', lw=2, zorder=1)
                line.set_path_effects(edge_path_effects)

    # Draw an arrow spanning width
    x1, y1 = node_positions[1][0]
    x2, y2 = node_positions[1][-1]
    ax.annotate('', xy=(x1 - spacing - 0.5, y1 + 0.25), xytext=(x1 - 3.5, y2 - 0.25),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2, mutation_scale=20, zorder=20))
    ax.text(x1 - spacing - 0.8, (y1+y2)/2, 'width', ha='center', va='center', rotation=90, fontsize=SIZE)

    # Draw an arrow spanning depth
    x1, y1 = node_positions[0][-1]
    x2, y2 = node_positions[-1][-1]
    ax.annotate('', xy=(x1 - 0.25, y1 - 1.45), xytext=(x2 +0.25, y1 - 1.45),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2, mutation_scale=20, zorder=20))
    ax.text((x1+x2)/2, y1-1.85, 'depth', ha='center', va='center', fontsize=SIZE)

    # Draw the plot
    ax.axis('off')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

# Define the layers of the neural network
layers = [3, 5, 5, 2]

# Draw the neural network in xkcd style with wider spacing, nodes on top, darker colors, adjusted edge thickness and node size, no title, transparent background, thinner black borders, numbers on nodes, and a large arrowhead pointing at node 2
draw_xkcd_neural_net(layers, spacing=3)
