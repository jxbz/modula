import matplotlib.pyplot as plt

# Enable xkcd style
plt.xkcd()

# Number of layers in the neural network
L = 9
mid_layer = 4
layers = [' W ' if i == mid_layer else r"$\quad$" for i in range(L)]

# Plot the layers horizontally with arrows and separate block for Delta W closer
fig, ax = plt.subplots(figsize=(12, 2.9))
fig.patch.set_alpha(0)  # Make the background transparent

# Create blocks for each layer and add arrows
for i, layer in enumerate(layers):
    if i < mid_layer:
        facecolor = '#FF6961'
    elif i == mid_layer:
        facecolor = '#FDFD96'
    else:
        facecolor = '#77B5FE'

    with plt.style.context({'path.effects': []}):
        ax.text(i-0.1, 0.8, layer, ha='center', va='center', fontsize=16, 
            bbox=dict(facecolor=facecolor, edgecolor='black', pad=10.0))
    if i < L - 1:
        ax.arrow(i, 0.8, 0.6, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')

# Highlight the middle layer with an update in a separate block above, closer

with plt.style.context({'path.effects': []}):
    ax.text(mid_layer-0.1, 1.8, 'ΔW', ha='center', va='center', fontsize=16,
        bbox=dict(facecolor='#FDFD96', edgecolor='black', pad=10.0))
ax.arrow(mid_layer-0.1, 1.8, 0, -0.70, head_width=0.1, head_length=0.1, fc='black', ec='black')

ax.text(1.5-0.1, 0.3, "head of the network", ha='center', va='center', fontsize=16)
ax.text(6.5-0.1, 0.3, "tail of the network", ha='center', va='center', fontsize=16)
ax.text(4-0.1, 0.3,   "middle layer", ha='center', va='center', fontsize=16)

# Show the plot
ax.axis('off')
ax.set_aspect('equal')
plt.tight_layout()
plt.show()
