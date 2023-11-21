import numpy as np
import matplotlib.pyplot as plt
from . import envs

from matplotlib.colors import LinearSegmentedColormap
def new_opacity_cmap(color, blend=True):
    '''
    This function returns a colormap that linearly varies the opacity of the supplied color for inputs.
    '''
    if len(color) == 4:
        assert color[-1] == 1
        color = color[:3]
    assert len(color) == 3, color
    if blend is True:
        blend = [1, 1, 1]
    if blend:
        rng = [
            tuple([
                c * opacity + b * (1 - opacity)
                for c, b in zip(color, blend)
            ]) + (1,)
            for opacity in [0, 1]
        ]
    else:
        rng = [color+(0.0,), color+(1.0,)]
    return LinearSegmentedColormap.from_list(f'OpacityColormap({color})', rng)


plot_cmap = new_opacity_cmap(plt.get_cmap('Greens')(0.8))


def plot(mdp, state=None, *, tidy=True, ax=None, scale=1., z=None):
    if ax is None:
        _, ax = plt.subplots()
    plt.sca(ax)
    if state is None:
        state = mdp.initial_state()

    # We invert loaded maps in parse_map(). If we weren't doing so, we'd need
    # the following: direction_to_rot = [90, 0, 270, 180]
    direction_to_rot = [270, 0, 90, 180]

    for xidx in range(mdp.map_h.shape[0]):
        for yidx in range(mdp.map_h.shape[1]):
            idx = mdp.map_light[xidx, yidx]
            if idx == envs.NO_LIGHT:
                continue
            fill = False
            if state.map_lit[idx] == envs.CONST_MAP_LIT_TRUE:
                fill = True
            ax.add_artist(plt.Circle((xidx, yidx), scale * 0.3, color='gold', fill=fill, linewidth=scale * 5))
    if z is None:
        z = mdp.map_h.T
    else:
        assert z.shape == mdp.map_h.T.shape
    plt.imshow(z, cmap=plot_cmap, vmin=np.min(z), vmax=np.max(z))
    #plt.annotate('>', position, rotation=direction_to_rot[direction])
    plt.annotate(
        '▶', state.position, rotation=direction_to_rot[state.direction],
        horizontalalignment='center', verticalalignment='center',
        fontsize=scale * 20, fontfamily='monospace')
    plt.colorbar()

    # Need to invert at the end...
    ax.invert_yaxis()

    if tidy:
        plt.xticks([])
        plt.yticks([])
        plt.gca().images[-1].colorbar.remove()

def plot_program(mdp, program, *, frames=False, ax=None, scale=1., plot_kw={}):
    (state, reward, limit), record_path = envs.interpret_program_record_path(mdp, program)

    if frames:
        assert ax is None
        rp = record_path[:-1] # skipping last one
        f, axes = plt.subplots(1, len(rp), figsize=(len(rp), 1))
        for ax, (state, action) in zip(axes, rp):
            plot(mdp, state, ax=ax, **plot_kw)
        return

    if ax is None:
        _, ax = plt.subplots()
    else:
        plt.sca(ax)
    pos_ = [state.position for state, i in record_path]
    steps = sum([1 for state, i in record_path if i is not None])
    plt.title(steps)
    plot(mdp, record_path[-1][0], ax=ax, scale=scale, **plot_kw)
    for prev, pos in zip(pos_[:-1], pos_[1:]):
        plt.plot([prev[0], pos[0]], [prev[1], pos[1]], c='k', lw=scale*1.5)

'''
def visualize_trajectory(map_, instructions):
    import matplotlib.pyplot as plt
    map_h, map_light, position, direction = map_
    map_lit = np.zeros(map_light.shape, dtype=np.int8)

    plt.figure()
    plt.title('Start')
    visualize_map((map_h, map_light, position, direction), map_lit=map_lit)

    for instruction in instructions:
        position, direction = interpret_instruction(
            map_h, map_light, position, direction, map_lit, instruction)
        plt.figure()
        plt.title(instruction.name)
        visualize_map((map_h, map_light, position, direction), map_lit=map_lit)

def visualize_static_trajectory(map_, instructions):
    import matplotlib.pyplot as plt
    map_h, map_light, position, direction = map_
    map_lit = np.zeros(map_light.shape, dtype=np.int8)

    pos_ = [position]
    for instruction in instructions:
        position, direction = interpret_instruction(
            map_h, map_light, position, direction, map_lit, instruction)
        pos_.append(position)
    plt.figure()
    visualize_map((map_h, map_light, position, direction), map_lit=map_lit)
    for prev, pos in zip(pos_[:-1], pos_[1:]):
        plt.plot([prev[0], pos[0]], [prev[1], pos[1]], c='k')

def visualize_program(map_, program, fig=None, tidy=False):
    import matplotlib.pyplot as plt
    map_h, map_light, position, direction = map_

    record_path = []
    position, direction, map_lit, _ = interpret_program(map_, program, record_path=record_path)
    pos_ = [p for p, d, i in record_path]
    steps = sum([1 for p, d, i in record_path if i is not None])
    fig or plt.figure()
    plt.title(steps)
    visualize_map((map_h, map_light, position, direction), map_lit=map_lit, tidy=tidy)
    for prev, pos in zip(pos_[:-1], pos_[1:]):
        plt.plot([prev[0], pos[0]], [prev[1], pos[1]], c='k')
'''


INSTRUCTION_TO_REPR = {
    'A': '💡',
    'B': 'J',
    'C': 'W',
    'D': 'R',
    'E': 'L',
    '1': 'P1',
    '2': 'P2',
    '3': 'P3',
    '4': 'P4',
}

def repr_instructions(pp):
    return ' '.join(INSTRUCTION_TO_REPR[i] for i in pp)

def repr_program(p, line_prefix=''):
    return line_prefix + (
        'M: '+repr_instructions(p.main) + '\n' +
        ''.join(
            line_prefix + f'{i+1}: ' + repr_instructions(sr) + '\n'
            for i, sr in enumerate(p.subroutines)
            if sr
        )
    ).strip()

def mpl_fmt(fn):
    import io
    import base64

    # Run the fn. We assume it sets cf and ca appropriately
    fn()

    # Save to string
    s = io.BytesIO()
    plt.savefig(s, format='png', bbox_inches="tight")
    # Make sure to close
    plt.close()
    # Return as base64
    s = base64.b64encode(s.getvalue()).decode("utf-8").replace("\n", "")
    return '<img src="data:image/png;base64,%s">' % s

def display_df_with_mpl(df, mpl_columns=[], *, to_html_kwargs={}):
    from IPython.display import display, HTML
    f = {k: mpl_fmt for k in mpl_columns}
    display(HTML(df.to_html(escape=False, formatters=f, **to_html_kwargs)))
