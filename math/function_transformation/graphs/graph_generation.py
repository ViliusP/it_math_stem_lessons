import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import shutil
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import Arrow

GRAPHS_DIR = './math/function_transformation/graphs'


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
        y_min, y_max = values_range if values_range else (min(Y_full), max(Y_full))
        ax.set_ylim(y_min, y_max)

        # Plot the main part of the function
        ax.plot(X_full, Y_full, label=label, color='blue')  # Ensure main plot is blue

        # Define Y-axis ticks
        y_ticks = np.arange(y_min, y_max + 1, y_tick_step if y_tick_step is not None else tick_step)

        # Add dashed lines at the start and end if include_dashed_lines is True
        if include_dashed_lines:
            # Find indices of the visible points
            visible_indices = np.where((Y_full >= y_min) & (Y_full <= y_max))[0]
            if len(visible_indices) > 0:
                n_points = 40  # Number of points for the dashed lines
                start_index = max(visible_indices[0] + n_points, 0)
                end_index = min(visible_indices[-1] - n_points, len(X_full) - 1)

                # Draw dashed lines for the first and last n visible points
                ax.plot(X_full[visible_indices[0]:start_index], Y_full[visible_indices[0]:start_index], linestyle='--', color='white', alpha=0.5)
                ax.plot(X_full[end_index:visible_indices[-1]], Y_full[end_index:visible_indices[-1]], linestyle='--', color='white', alpha=0.5)

        ax.set_yticks([tick for tick in y_ticks if y_min <= tick <= y_max and tick != 0])
        
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
        ax.set_xticks([tick for tick in np.arange(domain[0], domain[1] + 1, tick_step) if tick != 0])

        # Display a single zero at the origin
        ax.text(-.5, -.75, '0', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.1", edgecolor="none", facecolor="white"), zorder=2)

        # Add a legend with a border and adjust its position
        legend = ax.legend(loc='upper center', frameon=True,
                           bbox_to_anchor=(0.5, -0.025))
        legend.get_frame().set_edgecolor('black')

        # Save the figure before showing it
        plt.savefig(f'{GRAPHS_DIR}/{filename}', bbox_inches='tight')

# Pre
matplotlib.font_manager._load_fontmanager(try_read_cache=False)

# Generation start
# --------------------------------------------------
X = ["$x_1$", "$x_2$", "$x_3$", "$x_4$"]  # Including numbers and letters
Y = ["$y_1$", "$y_2$", "$y_3$", "$y_4$"]   # Example set with letters
f = {"$x_1$": "$y_2$", "$x_2$": "$y_1$", "$x_3$": "$y_3$",
     "$x_4$": "$y_4$"}  # Mapping between elements

plot_function_as_sets(X, Y, f, filename="functions_as_graphs.pdf")

X = [0, 1, 2, 3, 4, 5]  # Including numbers and letters
Y = [0, 1, 4, 9, 16, 25]   # Example set with letters
f = {0: 0, 1: 1, 2: 4, 3: 9, 4: 16, 5: 25}  # Mapping between elements

plot_function_as_sets(X, Y, f, filename="functions_as_graphs_example1.pdf")

# Example quadratics usage


def quadratic_formula(a=1, h=0, v=0):
    return lambda x: a * (x - h)**2 + v


plot_graph(domain=(-4, 4), func=quadratic_formula(),
           label='$f(x) = x^2$', filename='quadratic_function_plot_example.pdf')

plot_graph(domain=(-2, 6), func=quadratic_formula(h=2),
           label='$f(x) = (x-2)^2$', filename='quadratic_func_lsh_2.pdf')

plot_graph(domain=(-4, 4), func=quadratic_formula(v=2),
           label='$f(x) = x^2+2$', filename='quadratic_func_ush_2.pdf', values_range=(-2, 15))

plot_graph(domain=(-4, 4), func=quadratic_formula(a=-1),
           label='$f(x) = -x^2$', filename='quadratic_func_flipped.pdf', values_range=(-15, 2))

plot_graph(domain=(-4, 4), func=quadratic_formula(a=1/2),
           label='$f(x) = \\frac{1}{2}x^2$', filename='quadratic_func_shrink_2.pdf', values_range=(-2, 15))

# Example quadratics transformation usage
plot_graph(domain=(-4, 4), func=quadratic_formula(),
           label='$f(x) = x^2$', filename='quadratic_func_primary.pdf', values_range=(-2, 15), y_tick_step=2)

plot_graph(domain=(-4, 4), func=quadratic_formula(a=2),
           label='$2f(x)=2(x)^2$', filename='quadratic_func_transform_2.pdf',  values_range=(-2, 15), y_tick_step=2)

plot_graph(domain=(-4, 4), func=quadratic_formula(v=2),
           label='$f(x)+2 = x^2+2$', filename='quadratic_func_transform_3.pdf',  values_range=(-2, 15), y_tick_step=2)

plot_graph(domain=(-4, 4), func=quadratic_formula(a=.5),
           label='$\\frac{1}{2}f(x) = \\frac{1}{2}x^2$', filename='quadratic_func_transform_4.pdf',  values_range=(-2, 15), y_tick_step=2)

plot_graph(domain=(-4, 4), func=quadratic_formula(v=-2),
           label='$f(x)-2 = x^2-2$', filename='quadratic_func_transform_5.pdf',  values_range=(-2, 15), y_tick_step=2)

plot_graph(domain=(-4, 4), func=quadratic_formula(a=-1),
           label='$-f(x) = -x^2$', filename='quadratic_func_transform_6.pdf',  values_range=(-15, 2), y_tick_step=2)
