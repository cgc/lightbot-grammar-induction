'''
This file was copied over from the codebase used for the CCN 2018 submission. Originally by those authors, I've modified to adapt to this codebase.
'''

from itertools import product

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow, Circle
from matplotlib.path import Path
import matplotlib.patheffects as path_effects
import matplotlib.patches as patches

import numpy as np


class Compat:
    '''
    Compatibility layer added by CGC to use this code.
    '''
    @staticmethod
    def plot_program_nx(p):
        # DEPRECATED
        newp = {
            '5': p.main,
        } | {
            str(idx+1): p.subroutines[idx]
            for idx in range(4)
        }
        return plot_program_nx(newp)

    @staticmethod
    def visualize_trace(p, *args, **kwargs):
        from . import tools
        from . import program_analysis
        return visualize_trace(tools.readable_repr(program_analysis.FlattenProgram.flatten(p)), *args, **kwargs)

    @staticmethod
    def plot_program(p, *args, **kwargs):
        # This one was used for CCN 2018
        from . import tools
        p = tools.readable_repr(p)
        newp = {
            'program': p.main,
        } | {
            str(idx+1): p.subroutines[idx]
            for idx in range(4)
        }
        return plot_program(newp, *args, **kwargs)


def run_prog(prog, max_trace_len=100, get_execution_tree=False, main_pi=None):
    """
    Takes a program and returns a trace or execution trace.

    Parameters
    ----------
    prog : dict
        The program represented as a dictionary. Keys are
        integer-strings, values are programs, and the 'main'
        function is the highest valued integer.

    max_trace_len : int
        The longest number of characters to represent the trace

    get_execution_tree : bool
        Flag for whether to return a trace with parentheses to
        indicate the tree structure resulting from execution
        of the program.
    """
    if main_pi is None:
        main_pi = 'program'
    trace = prog[main_pi]
    num_in_trace = True
    while num_in_trace and len(trace) < max_trace_len:
        for sp in prog:
            if get_execution_tree:
                trace = trace.replace(sp, '(' + prog[sp] + ')')
            else:
                trace = trace.replace(sp, prog[sp])

        num_in_trace = False
        for sp in prog:
            if sp in trace:
                num_in_trace = True
                break
    return trace


ACTIONS = [c for c in 'ABCDERLJWS']


def get_program_execution_tree(prog, main_pi=None, max_trace_len=100):
    """
    This takes a program and recursively runs it while tracking the calling
    structure. It is useful for plotting and determining certain properties
    of the execution structure.

    :param prog: dict
        A program represented as a dictionary.

    :param main_pi: str
        The name of the main call in the program (i.e. a key).

    :param max_trace_len:
        The maximum length trace to roll out (only important for recursion)

    :return: dict
        A program tree represented as a dictionary will be returned. A program
        tree will have the keys: `exec`, the ordered execution of the program,
        including the sub-trees (themselves program trees) if a call was
        another subprocess; `max_depth`, the maximum depth of the calls from
        the current tree; `start_time`, the first timestep that this program
        or a child of this program is active; `end_time`, the last timestep;
        `depth`, the current depth from the root call.
    """
    if main_pi is None:
        main_pi = 'program'
    main_exec = [c for c in prog[main_pi]]
    timestep = [0, ]

    def expand_exec(execution, depth):
        max_depth = depth
        start_ts = timestep[0]

        for ci in range(len(execution)):
            c = execution[ci]
            if timestep[0] > max_trace_len:
                execution[ci] = (timestep[0], c)
                timestep[0] += 1
            elif c in ACTIONS:
                execution[ci] = (timestep[0], c)
                timestep[0] += 1
            else:
                child_exec = [c_ for c_ in prog[c]]
                exec_res = expand_exec(child_exec, depth + 1)
                exec_res['call'] = c
                execution[ci] = exec_res
                max_depth = max([max_depth, exec_res['max_depth']])

        end_ts = timestep[0] - 1

        return {
            'exec': execution,
            'max_depth': max_depth,
            'start_time': start_ts,
            'end_time': end_ts,
            'depth': depth
        }

    prog_tree = expand_exec(main_exec, 0)
    prog_tree['call'] = main_pi

    return prog_tree


# =========================================================================== #

#        Program execution tree visualization

# =========================================================================== #

def get_program_execution_tree_edges(prog_tree, parent_node_loc=.5):
    edges = []
    max_depth = prog_tree['max_depth'] + 1
    seen_calls = set()

    def get_edges_from_call(pt):
        reused = pt['call'] in seen_calls
        seen_calls.add(pt['call'])

        kw = {
            'parent_call': pt['call'],
            'reused': reused,
        }

        prog_x = (pt['end_time']) * parent_node_loc \
                 + pt['start_time'] * (1 - parent_node_loc)

        prog_loc = (prog_x, max_depth - pt['depth'])

        for c in pt['exec']:
            if isinstance(c, dict):
                child_x = (c['end_time']) * parent_node_loc \
                          + c['start_time'] * (1 - parent_node_loc)
                child_loc = (child_x, max_depth - c['depth'])
                edges.append({
                    'start': prog_loc,
                    'end': child_loc,
                    **kw
                })
                get_edges_from_call(c)
            else:
                ts, cn = c
                edges.append({
                    'start': prog_loc,
                    'end': (ts, max_depth - (pt['depth'] + 1)),
                    **kw
                })
                edges.append({
                    'start': (ts, max_depth - (pt['depth'] + 1)),
                    'end': (ts, 0),
                    **kw
                })

    get_edges_from_call(prog_tree)
    return edges


def plot_from_edges(edges, trace=None, yoffset=1, flip=False,
                    h_scale=3, scale=1, depth_exp=1,
                    main_name='program',
                    max_depth=4, process_colors=None, ax=None,
                    show_reuse=False,
                    ):
    # HACK: No need for yoffset b/c text is now positioned appropriately in data space
    yoffset = 0
    if trace is None:
        trace = '1234567890'
    if ax is None:
        fig, ax = plt.subplots(figsize=(len(trace) * .5 * scale,
                                        max_depth * .5 * h_scale * scale))
    if process_colors is None:
        # colors = [
        #     'mediumturquoise',
        #     'rosybrown',
        #     'forestgreen',
        #     'sandybrown',
        #     'darkorchid',
        #     'mediumblue',

        #     'teal',
        #     'firebrick',
        #     'green',
        #     'darkorange',
        #     'rebeccapurple',
        #     'royalblue',
        # ]
        colors = [
            '',
            'forestgreen',
            'darkorange',
            'firebrick',
            'darkorchid',
        ]
        process_colors = {str(i) : clr for i, clr in enumerate(colors)}
    # adjust edges as desired
    plot_edges = []
    if flip:
        flip = -1
    else:
        flip = 1

    def yt(y):
        return y ** depth_exp
    for edge in edges:
        sx, sy = edge['start']
        ex, ey = edge['end']
        plot_edges.append(edge | {
            'start': (sx, flip * ((yt(sy)) + yoffset)),
            'end': (ex, flip * ((yt(ey)) + yoffset)),
            # 'parent_call': edge['parent_call']
        })

    lw = 2
    for edge in plot_edges:
        s = edge['start']
        e = edge['end']
        if edge['parent_call'] == main_name:
            process_color = 'k'
            # linewidth = 4*.25*scale
            linewidth = lw*.5*scale
        else:
            c_ = str(int(edge['parent_call']) % len(process_colors))
            process_color = process_colors[c_]
            linewidth = lw*scale

        ax.plot([s[0], e[0]], [s[1], e[1]],
                linestyle='--' if show_reuse and edge['reused'] else '-',
                color=process_color,
                lw=linewidth)


def plot_program(prog, scale=1, h_scale=3, fontsize=18, depth_exp=1,
                 yoffset=.2, max_trace_len=100,
                 parent_node_loc=.5, top_down=True, show_reuse=False):
    trace = run_prog(prog, get_execution_tree=False)

    extree = get_program_execution_tree(prog, max_trace_len=max_trace_len)
    edges = get_program_execution_tree_edges(extree,
                                             parent_node_loc=parent_node_loc)
    max_depth = extree['max_depth']
    fig, ax = plt.subplots(
        frameon=False,
        figsize=(len(trace) * .5 * scale, max_depth * .5 * h_scale * scale)
    )
    ax.axis('off')
    kw = dict() if top_down else dict(flip=True)
    plot_from_edges(edges, trace=trace, yoffset=yoffset, ax=ax,
                    depth_exp=depth_exp, show_reuse=show_reuse, **kw)
    for ci, c in enumerate(trace):
        # ax.text(ci, 0, c, ha='center', va='center', size=fontsize * scale)
        ax.text(ci, -.1 * scale, c, ha='center', va='top', size=fontsize * scale, fontfamily='Courier')

    # HACK Not entirely sure why, but we always add horizontal padding. We remove it here.
    xlim = ax.get_xlim()
    ax.set_xlim((xlim[0] + .5, xlim[1] - .5))

    return fig


def plot_parse_comparison(prog1, prog2, scale=1, h_scale=3,
                          fontsize=18, depth_exp=1,
                          yoffset=.2, max_trace_len=100,
                          parent_node_loc=.5):
    trace = run_prog(prog1, get_execution_tree=False)

    extree1 = get_program_execution_tree(prog1,
                                         max_trace_len=max_trace_len)
    edges1 = get_program_execution_tree_edges(extree1,
                                              parent_node_loc=parent_node_loc)

    extree2 = get_program_execution_tree(prog2,
                                         max_trace_len=max_trace_len)
    edges2 = get_program_execution_tree_edges(extree2,
                                              parent_node_loc=parent_node_loc)
    max_depth1 = extree1['max_depth']
    max_depth2 = extree2['max_depth']

    max_depth = max([max_depth1, max_depth2])
    fig, ax = plt.subplots(figsize=(len(trace) * .5 * scale,
                                    2 * max_depth * .5 * h_scale * scale))
    ax.axis('off')
    plot_from_edges(edges1, trace=trace, yoffset=yoffset,
                    depth_exp=depth_exp, ax=ax)
    plot_from_edges(edges2, trace=trace, yoffset=yoffset,
                    ax=ax, depth_exp=depth_exp,
                    flip=True)
    for ci, c in enumerate(trace):
        ax.text(ci, 0, c, ha='center', va='center', size=fontsize * scale)
    return fig
# =========================================================================== #

#        Visualize path of a trace

# =========================================================================== #

def visualize_states(ax=None, states=None,
                     tile_color=None,
                     plot_size=None,
                     panels=None,
                     **kwargs):
    '''
        Supported kwargs:
            - tile_color : a dictionary from tiles (states) to colors
            - plot_size is an integer specifying how many tiles wide
              and high the plot is, with the grid itself in the middle
    '''
    if tile_color is None:
        tile_color = {}

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if panels is None:
        panels = []

    # plot squares
    for s in states:
        if s == (-1, -1):
            continue
        square = Rectangle(s, 1, 1, color=tile_color.get(s, 'white'), ec='k',
                           lw=2)
        ax.add_patch(square)

    ax.axis('off')
    if plot_size is None and len(panels) == 0:
        ax.set_xlim(-0.1, 1 + max([s[0] for s in states]) + .1)
        ax.set_ylim(-0.1, 1 + max([s[1] for s in states]) + .1)
        ax.axis('scaled')
    elif len(panels) > 0:
        xlim = [-0.1, 1 + max([s[0] for s in states]) + .1]
        ylim = [-0.1, 1 + max([s[1] for s in states]) + .1]
        if 'right' in panels:
            xlim[1] += 2
        if 'left' in panels:
            xlim[0] -= 2
        if 'top' in panels:
            ylim[1] += 2
        if 'bottom' in panels:
            ylim[0] -= 2
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    else:
        cx = (max([s[0] for s in states])+1)/2
        cy = (max([s[1] for s in states])+1)/2
        ax.set_xlim(cx-0.1-plot_size/2, cx+0.1+plot_size/2)
        ax.set_ylim(cy-0.1-plot_size/2, cy+0.1+plot_size/2)
    return ax


def visualize_trace(trace, ax=None, jitter_mean=0, jitter_var=0,
                    scale=1):
    if ax is None:
        fig = plt.figure(figsize=(scale * 10, scale * 10))
        ax = fig.add_subplot(111)
    head_to_action = {
        (0, 1): '^',
        (1, 0): '>',
        (-1, 0): '<',
        (0, -1): 'v'
    }

    # run through the trajectory to get the states visited
    traj = []
    head = (0, 1)
    s = (0, 0)
    max_xy = (-np.inf, -np.inf)
    min_xy = (np.inf, np.inf)
    for a in trace:
        if a == 'W':
            ns = tuple(np.add(s, head))
            traj.append((s, a, ns))
            s = ns
        elif a == 'J':
            ns = tuple(np.add(s, head))
            traj.append((s, a, ns))
            s = ns
        elif a == 'L':
            head = tuple(np.dot([[0, -1], [1, 0]], head))
            traj.append((s, a, s))
        elif a == 'R':
            head = tuple(np.dot([[0, 1], [-1, 0]], head))
            traj.append((s, a, s))
        elif a == 'S':
            traj.append((s, a, s))

        if traj[-1] is not None:
            max_xy = np.maximum(max_xy, traj[-1][2])
            max_xy = np.maximum(max_xy, traj[-1][0])
            min_xy = np.minimum(min_xy, traj[-1][2])
            min_xy = np.minimum(min_xy, traj[-1][0])

    # get translated traj and a plotting traj
    plot_traj = []
    new_traj = []
    for s, a, ns in traj:
        new_s = tuple(np.add(s, np.negative(min_xy)))
        new_ns = tuple(np.add(ns, np.negative(min_xy)))
        new_traj.append((new_s, a, new_ns))
        if a in 'LRS':
            continue
        plot_traj.append((new_s, a, new_ns))
    plot_traj.append((new_traj[-1][2], 'x', None))
    traj = new_traj

    # identify switch locations and jump locations
    switch_locs = []
    min_heights = {}
    max_min_height = 0
    height = 0
    higher_locs = []
    lower_locs = []
    for s, a, ns in traj:
        if a == 'S':
            switch_locs.append(s)
        elif a == 'J':
            lower_locs.append(s)
            higher_locs.append(ns)
            if s in min_heights:
                min_heights[s] = max([min_heights[s], height])
            else:
                min_heights[s] = height
            height += 1
            if ns in min_heights:
                min_heights[ns] = max([min_heights[ns], height])
            else:
                min_heights[ns] = height
            max_min_height = max([max_min_height, height])
        elif a == 'W':
            height = 0

    # plot states and 'higher tiles' as grey
    tile_colors = {}
    for s, h in min_heights.items():
        tile_colors[s] = tuple([1 - .5*h/max_min_height,]*3)
    states = list(product(range(int(max_xy[0] - min_xy[0]) + 1),
                          range(int(max_xy[1] - min_xy[1]) + 1)))
    visualize_states(ax=ax, states=states, tile_color=tile_colors)

    # plot switch locations with Xs
    for s in switch_locs:
        light_text = ax.text(
            s[0] + .5, s[1] + .5, 'x',
            fontsize=scale * 50, color='green',
            ha='center', va='center'
        )
    visualize_trajectory(axis=ax, traj=plot_traj, jitter_var=jitter_var,
                         jitter_mean=jitter_mean,
                         lw=3)
    ax.text(plot_traj[0][0][0] + .25, plot_traj[0][0][1] + .25, 'Start',
            fontsize=scale * 15, ha='center', va='center')
    ax.text(plot_traj[-1][0][0] + .25, plot_traj[-1][0][1] + .25, 'End',
            fontsize=scale * 15, ha='center', va='center')

    return fig


def visualize_trajectory(axis=None, traj=None,
                         jitter_mean=0,
                         jitter_var=.1,
                         plot_actions=False,
                         endpoint_jitter=False,
                         color='black',
                         outline=False,
                         outline_color='white',
                         lw=1,
                         **kwargs):

    traj = [(t[0], t[1]) for t in traj] #traj only depends on state actions

    if len(traj) == 2:
        p0 = tuple(np.array(traj[0][0]) + .5)
        p2 = tuple(np.array(traj[1][0]) + .5)
        p1 = np.array([(p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2]) \
                        + np.random.normal(0, jitter_var, 2)
        if endpoint_jitter:
            p0 = tuple(
                np.array(p0) + np.random.normal(jitter_mean, jitter_var, 2))
            p1 = tuple(
                np.array(p1) + np.random.normal(jitter_mean, jitter_var, 2))
        segments = [[p0, p1, p2], ]
    elif (len(traj) == 3) and (traj[0][0] == traj[2][0]):
        p0 = tuple(np.array(traj[0][0]) + .5)
        p2 = tuple(np.array(traj[1][0]) + .5)
        if abs(p0[0] - p2[0]) > 0:  # horizontal
            jitter = np.array(
                [0, np.random.normal(jitter_mean, jitter_var * 2)])
            p2 = p2 - np.array([.25, 0])
        else:  # vertical
            jitter = np.array(
                [np.random.normal(jitter_mean, jitter_var * 2), 0])
            p2 = p2 - np.array([0, .25])
        p1 = p2 + jitter
        p3 = p2 - jitter
        segments = [[p0, p1, p2], [p2, p3, p0]]
    else:
        state_coords = []
        for s, a in traj:
            jitter = np.random.normal(jitter_mean, jitter_var, 2)
            coord = np.array(s) + .5 + jitter
            state_coords.append(tuple(coord))
        if not endpoint_jitter:
            state_coords[0] = tuple(np.array(traj[0][0]) + .5)
            state_coords[-1] = tuple(np.array(traj[-1][0]) + .5)
        join_point = state_coords[0]
        segments = []
        for i, s in enumerate(state_coords[:-1]):
            ns = state_coords[i + 1]

            segment = []
            segment.append(join_point)
            segment.append(s)
            if i < len(traj) - 2:
                join_point = tuple(np.mean([s, ns], axis=0))
                segment.append(join_point)
            else:
                segment.append(ns)
            segments.append(segment)

    outline_patches = []
    if outline:
        for segment, step in zip(segments, traj[:-1]):
            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            path = Path(segment, codes)
            outline_patch = patches.PathPatch(path, facecolor='none',
                                              capstyle='butt',
                                              edgecolor=outline_color, lw=lw*2)
            if axis is not None:
                axis.add_patch(outline_patch)
            outline_patches.append(outline_patch)

    traj_patches = []
    action_patches = []
    for segment, step in zip(segments, traj[:-1]):
        state = step[0]
        action = step[1]

        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(segment, codes)

        patch = patches.PathPatch(path, facecolor='none', capstyle='butt',
                                  edgecolor=color, lw=lw, **kwargs)
        traj_patches.append(patch)
        if axis is not None:
            axis.add_patch(patch)

        if plot_actions:
            dx = 0
            dy = 0
            if action == '>':
                dx = 1
            elif action == 'v':
                dy = -1
            elif action == '^':
                dy = 1
            elif action == '<':
                dx = -1
            action_arrow = patches.Arrow(segment[1][0], segment[1][1],
                                         dx*.4,
                                         dy*.4,
                                         width=.25,
                                         color='grey')
            action_patches.append(action_arrow)
            if axis is not None:
                axis.add_patch(action_arrow)
    return {
        'outline_patches': outline_patches,
        'traj_patches': traj_patches,
        'action_patches': action_patches
    }

# =========================================================================== #

#        Visualization using networkx (based on Sophia's code)

# =========================================================================== #


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def program_to_tree(prog):
    import networkx as nx

    main_i = max(prog.keys())
    root = [(main_i + '.0', x + '.' + str(i + 1)) for i, x in
            enumerate(prog[main_i])]
    G = nx.DiGraph()
    G.add_edges_from(root)
    terminal_nodes = [x for x in G.nodes() if
                      G.out_degree(x) == 0 and G.in_degree(x) == 1]
    unbranched = [x for x in terminal_nodes if isfloat(x)]
    while len(unbranched) > 0:
        for n in unbranched:
            if isfloat(n):
                start_n = len(G.nodes)
                if n[0] in prog:
                    for i, x in enumerate(prog[n[0]]):
                        if x[0] in prog:
                            if n[0] not in prog[x[0]] and n[0] != x[0]:
                                G.add_edge(n, x + '.' + str(i + start_n))
                        else:
                            G.add_edge(n, x + '.' + str(i + start_n))
                else:
                    G.remove_node(n)
                terminal_nodes = [x for x in G.nodes() if
                                  G.out_degree(x) == 0 and \
                                  G.in_degree(x) == 1]
                unbranched = [x for x in terminal_nodes if isfloat(x)]
    return list(G.edges())

def plot_program_nx(prog):
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout

    tree_data = program_to_tree(prog)

    color_dic = {'A': '#ffcc33', 'B':'#a9ec93', 'C': '#52e0bd',
                 'D': '#ff794d', 'E': '#ff794d',
                 '0': '#cce0ff', '1': '#66a3ff', '2': '#005ce6',
                 '3': '#003380', '4': '#00264d',
                 '5': '#FF00E4'}

    G = nx.DiGraph(tree_data)
    G.remove_node('5.0')

    labels = {}
    colors = []
    for n in G.nodes():
        labels[n] = n[0]
        colors.append(color_dic[n[0]])
    edge_colors = []
    for src, dest in G.edges():
        sr, _ = src.split('.')
        edge_colors.append({
            '1': 'forestgreen',
            '2': 'darkorange',
            '3': 'firebrick',
            '4': 'darkorchid',
        }[sr])
    pos = graphviz_layout(G, prog=None)
    nx.draw(
        G,
        pos,
        with_labels=False,
        node_color='white',
        edge_color=edge_colors,
        width=3,
        arrows=False)
    fig = nx.draw_networkx_labels(G, pos, labels,
                                  font_size=12, font_color="black")
    return fig
