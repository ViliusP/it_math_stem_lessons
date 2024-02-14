import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arrow
import matplotlib.cm as cm

GRAPHS_DIR = './math/graph_drawing'


def gen_example_xkcd():
    # Now we can load pyplot and apply xkcd style within its context
    with plt.xkcd():
        plt.rcParams['font.family'] = 'Uberhand Text Pro'

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        plt.xticks([])
        plt.yticks([])
        ax.set_ylim([-30, 10])

        data = np.ones(100)
        data[70:] -= np.arange(30)

        plt.annotate(
            'THE DAY I REALIZED\nI COULD COOK BACON\nWHENEVER I WANTED',
            xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))

        plt.plot(data)

        plt.xlabel('time')
        plt.ylabel('my overall health')

        # Save the figure before showing it
        fig.savefig(f'{GRAPHS_DIR}/histogram.pdf', bbox_inches='tight')


def plot_function_as_sets(X, Y, f, filename: str):
    with plt.xkcd():
        plt.rcParams['font.family'] = 'Uberhand Text Pro'

        fig, ax = plt.subplots()

        # Determine the heights for each set
        set_height_X = list(range(len(X)))
        set_height_Y = list(range(len(Y)))
        max_set_height = max(
            set_height_X[-1] if X else 0, set_height_Y[-1] if Y else 0)

        # Adjust vertical limit to add space above the highest element
        vertical_limit = max_set_height + 1

        # Draw sets X and Y as scatter plots along the vertical borders
        scatter_x = ax.scatter([0]*len(X), set_height_X, s=100, c='blue',
                               label='Aibė X', edgecolors='black', linewidths=1, marker='o')
        scatter_y = ax.scatter([2]*len(Y), set_height_Y, s=100, c='red',
                               label='Aibė Y', edgecolors='black', linewidths=1, marker='o')

        # Draw a border around the sets
        border_padding = 0.5
        ax.add_patch(patches.Rectangle((-border_padding, -border_padding),
                     2 * border_padding, vertical_limit, fill=False, edgecolor='black'))
        ax.add_patch(patches.Rectangle((2 - border_padding, -border_padding),
                     2 * border_padding, vertical_limit, fill=False, edgecolor='black'))

        # Label elements of sets X and Y
        for i, x in enumerate(X):
            ax.text(-0.2, set_height_X[i], str(x), ha='right', va='center')

        for i, y in enumerate(Y):
            ax.text(2.2, set_height_Y[i], str(y), ha='left', va='center')

        # Draw arrows to represent the function mapping
        for x, y in f.items():
            ax.annotate("", xy=(2, set_height_Y[Y.index(y)]), xytext=(0, set_height_X[X.index(x)]),
                        arrowprops=dict(arrowstyle="->", lw=1.0))

        ax.spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1, 3)
        ax.set_ylim(-1, vertical_limit)

        # Add a legend with a border and adjust its position
        legend = ax.legend(handles=[
                           scatter_x, scatter_y], loc='upper center', frameon=True, bbox_to_anchor=(0.5, 1.1))
        legend.get_frame().set_edgecolor('black')

        # Save the figure before showing it
        plt.savefig(f'{GRAPHS_DIR}/{filename}',
                    bbox_inches='tight')


def plot_graph(domain, func, label, filename: str, tick_step=2, values_range=None, y_tick_step=None, include_dashed_lines=True):
    with plt.xkcd():
        plt.rcParams['font.family'] = 'Uberhand Text Pro'

        fig, ax = plt.subplots()

        # Generate X values from the domain and compute Y values using the function
        X_full = np.linspace(domain[0], domain[1], 1000)
        Y_full = func(X_full)

        # Determine the visible range based on values_range
        y_min, y_max = values_range if values_range else (
            min(Y_full), max(Y_full))
        ax.set_ylim(y_min, y_max)

        # Plot the main part of the function
        # Ensure main plot is blue
        ax.plot(X_full, Y_full, label=label, color='blue')

        # Define Y-axis ticks
        y_ticks = np.arange(
            y_min, y_max + 1, y_tick_step if y_tick_step is not None else tick_step)

        # Add dashed lines at the start and end if include_dashed_lines is True
        if include_dashed_lines:
            # Find indices of the visible points
            visible_indices = np.where(
                (Y_full >= y_min) & (Y_full <= y_max))[0]
            if len(visible_indices) > 0:
                n_points = 40  # Number of points for the dashed lines
                start_index = max(visible_indices[0] + n_points, 0)
                end_index = min(
                    visible_indices[-1] - n_points, len(X_full) - 1)

                # Draw dashed lines for the first and last n visible points
                ax.plot(X_full[visible_indices[0]:start_index], Y_full[visible_indices[0]                        :start_index], linestyle='--', color='white', alpha=0.5)
                ax.plot(X_full[end_index:visible_indices[-1]],
                        Y_full[end_index:visible_indices[-1]], linestyle='--', color='white', alpha=0.5)

        ax.set_yticks([tick for tick in y_ticks if y_min <=
                      tick <= y_max and tick != 0])

        # Move left y-axis and bottom x-axis to zero
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set aspect ratio to equal to make the graph equally even
        ax.set_aspect('equal')

        # Make arrows with labels
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
        ax.text(1, 1, 'X', transform=ax.get_yaxis_transform(),
                ha='center', va='center')

        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.text(1, 1, 'Y', transform=ax.get_xaxis_transform(),
                ha='center', va='center')

        # Set X-axis ticks using the same tick step, exclude zero
        ax.set_xticks([tick for tick in np.arange(
            domain[0], domain[1] + 1, tick_step) if tick != 0])

        # Display a single zero at the origin
        ax.text(-.5, -.75, '0', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.1", edgecolor="none", facecolor="white"), zorder=2)

        # Add a legend with a border and adjust its position
        legend = ax.legend(loc='upper center', frameon=True,
                           bbox_to_anchor=(0.5, -0.025))
        legend.get_frame().set_edgecolor('black')

        # Save the figure before showing it
        plt.savefig(f'{GRAPHS_DIR}/{filename}', bbox_inches='tight')


def plot_combined_lines(points, filename):
    with plt.xkcd():
        plt.rcParams.update({
            'grid.linewidth': 0.5,
            'axes.grid': True,
        })

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Get a colormap
        cmap = cm.get_cmap('Blues')  # Sequential colormap

        # Calculate the number of line segments
        num_segments = len(points) - 1

        # Plot each line segment with different shades
        for i in range(num_segments):
            x_values = [points[i][0], points[i + 1][0]]
            y_values = [points[i][1], points[i + 1][1]]
            # Adjust color selection to avoid very light colors
            # Adjust range within colormap
            color = cmap(0.7 + 0.4 * i / num_segments)
            ax.plot(x_values, y_values, marker='o', color=color)

        # Adjust axes and ticks
        ax.set_xlim(-2, 4)  # Extend x-axis
        ax.set_ylim(-3, 3)  # Extend y-axis

        # Set same spacing for both axes
        ax.set_aspect('equal', adjustable='box')

        # Set ticks to step of 1, remove zero, and the largest ticks
        x_ticks = [tick for tick in ax.get_xticks() if tick != 0 and tick != 4]
        y_ticks = [tick for tick in ax.get_yticks() if tick != 0 and tick != 3]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        # Add zero label near the origin
        ax.text(-0.1, -0.15, '0', ha='right', va='top')

        # Add arrowheads
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)

        # Move left y-axis and bottom x-axis to zero
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Save the plot
        plt.savefig(f'{GRAPHS_DIR}/{filename}', bbox_inches='tight')


def plot_quadratic_functions():
    with plt.xkcd():
        # Update the style for the grid
        plt.rcParams.update({
            'grid.linewidth': 0.5,
            'axes.grid': True,
        })

        # Define the primary quadratic function f(x) = x^2
        def f(x):
            return (x-3)**2-1

        # Define the translated quadratic function g(x) = (x + 2)^2 - 3
        def g(x):
            return -((x + 3)**2)+1

     # Create the figure and axis
        fig, ax = plt.subplots()

        # Generate x values for both functions
        x_values = np.linspace(-6, 6, 400)

        # Plot the primary quadratic function f(x)
        ax.plot(x_values, f(x_values), label='f(x)', color='blue')

        # Plot the translated quadratic function g(x)
        ax.plot(x_values, g(x_values), label='g(x)', color='orange')

        # Improve label positioning
        ax.text(2, f(2)+0.5, 'f(x)', fontsize=12, verticalalignment='bottom', horizontalalignment='left', color='blue')
        ax.text(-5, g(-2)+0.5, 'g(x)', fontsize=12, verticalalignment='bottom', horizontalalignment='left', color='orange')

        # Set ticks to step of 1, remove zero, and the largest ticks
        ax.set_xticks([tick for tick in np.arange(-6, 7, 1) if tick != 0])
        ax.set_yticks([tick for tick in np.arange(-5, 6, 1) if tick not in [0, 5]])

        # Set aspect of the plot to be equal for both axes
        ax.set_aspect('equal', adjustable='box')

        # Set limits for x and y axes
        ax.set_xlim(-6, 6)
        ax.set_ylim(-5, 5)

        # Make arrows with labels
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
        ax.text(1, 1, 'X', transform=ax.get_yaxis_transform(),
                ha='center', va='center')

        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.text(1, 1, 'Y', transform=ax.get_xaxis_transform(),
                ha='center', va='center')

        # Move left y-axis and bottom x-axis to zero
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Add grid
        ax.grid(True, which='both')

        # Save the plot to a file
        plt.savefig(f'{GRAPHS_DIR}/exercise_6_two_functions.pdf', bbox_inches='tight')


# Convert degrees to radians for plotting
def plot_sine_wave():
    with plt.xkcd():
        plt.rcParams.update({
            'grid.linewidth': 0.5,
            'axes.grid': True,
        })
        
        # Generate x values from -90° to 360° in degrees, convert to radians
        x_degrees = np.linspace(-360, 450, 1000)
        x_radians = x_degrees * np.pi / 180  # Convert degrees to radians
        y_values = np.sin(x_radians)

        # Create the figure and axis
        fig, ax = plt.subplots()

        # Plot the sine wave
        ax.plot(x_degrees, y_values, label='$f(x) = \sin x$')

        # Set ticks to step of 90, remove zero, and the largest ticks
        x_ticks = [-360, -270,-180,-90, 90, 180, 270, 360]
        y_ticks = [-1, 0, 1]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f'{int(tick)}°' for tick in x_ticks], fontsize=10)
        ax.set_yticks(y_ticks)

        # Set limits for x and y axes
        ax.set_xlim(-360, 450)
        ax.set_ylim(-1.25, 1.25)

        # Move left y-axis and bottom x-axis to zero
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Make arrows with labels
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
        ax.text(1, .125, 'X', transform=ax.get_yaxis_transform(),
                ha='center', va='center')

        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.text(20, 1, 'Y', transform=ax.get_xaxis_transform(),
                ha='center', va='center')

        ax.text(250, 1, '$f(x) = \sin x$', fontsize=14, verticalalignment='center', horizontalalignment='left')

        # Add grid
        ax.grid(True, which='both')

        # Save the plot to a file
        plt.savefig(f'{GRAPHS_DIR}/exercise_8_sin_wave.pdf', bbox_inches='tight')
       

def plot_sideways_parabola_xkcd(a, h, k, x_range, filename):
    """
    Plots a sideways parabola of the form (y - k)^2 = 4a(x - h) in xkcd style,
    where (h, k) is the vertex of the parabola.
    
    Parameters:
    - a: The parameter that defines the parabola's openness and direction.
    - h: The x-coordinate of the parabola's vertex.
    - k: The y-coordinate of the parabola's vertex.
    - x_range: A tuple (x_min, x_max) defining the range of x values for plotting.
    - filename: The name of the file to save the plot.
    """
        # Generate x values more densely around the vertex to avoid empty space
    x_values = np.linspace(x_range[0], x_range[1], 1000)
    y_values_positive = np.sqrt(4 * a * (x_values - h)) + k
    y_values_negative = -np.sqrt(4 * a * (x_values - h)) + k

    # Plotting in xkcd style
    with plt.xkcd():
        fig, ax = plt.subplots()
        ax.plot(x_values, y_values_positive, 'b')  # Positive branch
        ax.plot(x_values, y_values_negative, 'b')  # Negative branch
       
           # Move left y-axis and bottom x-axis to zero
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Set aspect ratio to equal to make the graph equally even
        ax.set_aspect('equal')
     
        # Make arrows with labels
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False, zorder=5)
        ax.text(1.1, -0.05, 'X', transform=ax.get_yaxis_transform(),
                ha='right', va='top')

        ax.plot((0), (1), ls="", marker="^", ms=10, color="k",
                transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
        ax.text(-0.05, 1.1, 'Y', transform=ax.get_xaxis_transform(),
                ha='right', va='top')

        # Add zero label near the origin
        ax.text(-0.1, -0.15, '0', ha='right', va='top')
        
        # Save the plot
        plt.savefig(f'{GRAPHS_DIR}/{filename}', bbox_inches='tight')
        plt.show()

# Example usage


# Pre
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

points = [(-1, -2), (0, 1), (1, 0), (2, 1), (3, 0)]

# Exercises
# 4
plot_combined_lines(points, 'f_function_test_1_1.pdf')
plot_combined_lines(points, 'f_function_test_1_2.pdf')

# 6

plot_sideways_parabola_xkcd(a=2, h=0, k=0, x_range=(-10, 10), filename='sideways_parabola_function_test_1.pdf')
