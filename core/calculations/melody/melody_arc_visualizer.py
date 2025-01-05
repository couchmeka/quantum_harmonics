import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class ArcDiagram:
    def __init__(self, nodes=None, title="Resonance Network"):
        """Initialize arc diagram with nodes and title"""
        self.nodes = nodes if nodes else []
        self.title = title
        self.connections = []
        self.fig = None
        self.ax = None
        self.bg_color = "white"
        self.cmap = "viridis"

    def set_background_color(self, color):
        """Set background color of the diagram"""
        self.bg_color = color

    def set_color_map(self, cmap):
        """Set colormap for the connections"""
        self.cmap = cmap

    def connect(self, source, target, weight=1):
        """Add a connection between nodes"""
        self.connections.append({"source": source, "target": target, "weight": weight})

    def _calculate_node_positions(self):
        """Calculate horizontal positions for nodes"""
        unique_nodes = list(set(self.nodes))
        positions = {}
        width = len(unique_nodes) - 1

        for i, node in enumerate(unique_nodes):
            if width == 0:
                positions[node] = 0
            else:
                positions[node] = i / width

        return positions

    def show_plot(self):
        """Display the arc diagram"""
        # Create figure with better size
        plt.close("all")
        self.fig, self.ax = plt.subplots(figsize=(12, 6))

        # Set background color
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)

        # Calculate node positions
        pos = self._calculate_node_positions()

        # Get color map
        color_map = plt.get_cmap(self.cmap)

        # Draw connections
        for conn in self.connections:
            start = pos[conn["source"]]
            end = pos[conn["target"]]

            # Calculate arc height based on distance
            height = 0.2 + abs(end - start) * 0.3

            # Create points for the arc
            x = np.linspace(start, end, num=50)
            y = height * np.sin(np.pi * (x - start) / (end - start))

            # Draw the arc
            color = color_map(conn["weight"])
            self.ax.plot(x, y, color=color, linewidth=2, alpha=0.7)

        # Draw nodes
        node_y = np.zeros(len(pos))
        node_x = [pos[node] for node in pos.keys()]
        self.ax.scatter(node_x, node_y, s=100, c="white", edgecolor="black", zorder=100)

        # Add node labels
        for node, x in pos.items():
            self.ax.annotate(
                str(node),
                (x, 0),
                xytext=(0, -20),
                textcoords="offset points",
                ha="center",
                va="top",
                bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
            )

        # Set title
        self.ax.set_title(self.title, pad=20, fontsize=14)

        # Set limits and remove axes
        self.ax.set_xlim(-0.1, 1.1)
        self.ax.set_ylim(-0.2, 0.5)
        self.ax.axis("off")

        # Adjust layout
        plt.tight_layout()
        return self.fig


def create_arc_diagram(
    data,
    source_col,
    target_col,
    weight_col=None,
    bg_color="white",
    cmap="viridis",
    fig=None,
    ax=None,
):
    """Create an arc diagram visualizing musical note relationships

    Parameters:
        data: DataFrame containing source, target nodes, and connection information
        source_col: Column name for source nodes
        target_col: Column name for target nodes
        weight_col: Column name for connection weights (ratio)
        bg_color: Background color of the plot
        cmap: Colormap for connections
        fig: Matplotlib figure (optional)
        ax: Matplotlib axis (optional)
    """
    # Get all unique nodes
    nodes = pd.concat([data[source_col], data[target_col]]).unique()

    # Create figure if not provided
    if fig is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    elif ax is None:
        ax = fig.add_subplot(111)

    # Set background color
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Calculate node positions
    pos = {}
    width = len(nodes) - 1
    for i, node in enumerate(nodes):
        if width == 0:
            pos[node] = 0
        else:
            pos[node] = i / width

    # Get color map
    color_map = plt.get_cmap(cmap)

    # Draw connections
    for _, row in data.iterrows():
        start = pos[row[source_col]]
        end = pos[row[target_col]]
        weight = row[weight_col] if weight_col else 1

        # Calculate arc height based on distance
        height = 0.15 + abs(end - start) * 0.25  # Reduced height

        # Create points for the arc
        x = np.linspace(start, end, num=50)
        y = height * np.sin(np.pi * (x - start) / (end - start))

        # Draw the arc
        color = color_map(weight)
        ax.plot(x, y, color=color, linewidth=2, alpha=0.7)

        # Add ratio information above the arc - adjusted position
        midpoint_x = (start + end) / 2
        midpoint_y = height + 0.01  # Reduced offset

        # Get shared elements information if available
        elements_text = ""
        if "common_elements" in row and row["common_elements"]:
            elements_text = "\nShared: " + ", ".join(
                row["common_elements"][:2]
            )  # Limit to 2 elements

        label = f"Ratio: {weight:.3f}{elements_text}"

        ax.annotate(
            label,
            (midpoint_x, midpoint_y),
            ha="center",
            va="bottom",
            bbox=dict(
                facecolor="white",
                edgecolor=color,
                alpha=0.8,
                boxstyle="round,pad=0.2",  # Reduced padding
            ),
            fontsize=7,  # Slightly smaller font
        )

    # Draw nodes
    node_y = np.zeros(len(pos))
    node_x = [pos[node] for node in pos.keys()]
    ax.scatter(node_x, node_y, s=100, c="white", edgecolor="black", zorder=100)

    # Add node labels with frequency information
    # for node, x in pos.items():
    #     # Extract note name and frequency from the node label
    #     if isinstance(node, str) and "\n" in node:
    #         note_name, freq = node.split("\n")
    #         label = f"{note_name}\n{freq}"
    #     else:
    #         label = str(node)
    #
    #     # Create a more visible label
    #     ax.annotate(
    #         label,
    #         (x, 0),
    #         xytext=(0, 20),  # Changed from -20 to 20 to move up
    #         textcoords="offset points",
    #         ha="center",
    #         va="bottom",  # Changed from 'top' to 'bottom'
    #         bbox=dict(
    #             boxstyle="round,pad=0.5",
    #             fc="white",
    #             ec="black",
    #             alpha=1.0,
    #             mutation_scale=1.0,
    #         ),
    #         fontsize=9,
    #         fontweight="bold",
    #         zorder=101,
    #     )
    for node, x in pos.items():
        parts = node.split("\n")
        note_name = parts[0]
        freq = parts[1]
        element_info = "\n".join(parts[2:]) if len(parts) > 2 else ""

        label = f"{note_name}\n{freq}"
        if element_info:
            label += f"\n{element_info}"

        ax.annotate(
            label,
            (x, 0),
            xytext=(0, 20),
            textcoords="offset points",
            ha="center",
            va="bottom",
            bbox=dict(
                boxstyle="round,pad=0.5",
                fc="white",
                ec="black",
                alpha=1.0,
                mutation_scale=1.0,
            ),
            fontsize=9,
            fontweight="bold",
            zorder=101,
        )

    # Set limits with tighter bounds
    ax.set_xlim(-0.05, 1.05)  # Reduced margins
    ax.set_ylim(-0.35, 0.55)  # Adjusted to prevent bleeding
    ax.axis("off")

    # Adjust layout with tighter spacing
    plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.15)

    # Use tight_layout with specific padding
    fig.tight_layout(pad=1.0)  # Adjust pad value as needed

    # Adjust layout
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
    return fig
