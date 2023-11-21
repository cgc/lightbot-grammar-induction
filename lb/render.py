import os
import types
import math
import pathlib

import matplotlib.pyplot as plt
import matplotlib as mpl
import PIL.Image
import PIL.ImageDraw

from . import envs

DEFAULT_SIZE = (690, 670)
CODE_DIR = pathlib.Path(__file__).absolute().parent
SPRITES = PIL.Image.open(CODE_DIR / 'assets/sprites.png').convert('RGBA')

def hex_to_color(hex):
    return PIL.ImageColor.getcolor(hex, 'RGB')

COLORS = types.SimpleNamespace()
COLORS.top = hex_to_color("#c9d3d9")
COLORS.front = hex_to_color("#adb8bd")
COLORS.side = hex_to_color("#e5f0f5")
COLORS.stroke = hex_to_color("#485256")
COLORS.top_light_on = hex_to_color("#FFE545")
COLORS.top_light_on_overlay = hex_to_color("#FEFBAF")
COLORS.top_light_off = hex_to_color("#0468fb")
COLORS.top_light_off_overlay = hex_to_color("#4c81ff")
COLORS.trajectory = (7, 192, 195)
# COLORS.trajectory = (241, 202, 25)
WHITE = (255, 255, 255)

class LBRPIL:
    '''
    Originally wrote this using PIL, but switched for vector graphics.
    Refactored out a small PIL-like interface that is reimplemented below in matplotlib.

    At some point, I considered a variant that would anti-alias (to avoid jagged lines) by
    rescaling all coordinates. That's been left out of this version.
    '''
    img: PIL.Image.Image
    ctx: PIL.ImageDraw.ImageDraw

    def __init__(self, img):
        self.img = img
        self.ctx = PIL.ImageDraw.Draw(img)
        self.width = self.img.width
        self.height = self.img.height

    @classmethod
    def new(cls, *, size=DEFAULT_SIZE):
        img = PIL.Image.new('RGBA', size, WHITE)
        return cls(img)

    def save(self, fn):
        self.img.save(fn)

    def polygon(self, pts, c, *args):
        if args:
            s, sw = args
            # Since this doesn't support fractional linewidths, we round up here.
            sw = int(math.ceil(sw))
            args = s, sw
        return self.ctx.polygon(pts, c, *args)

    def alpha_composite(self, im, dest):
        return self.img.alpha_composite(im, dest)

    def _repr_png_(self):
        return self.img._repr_png_()

    def show(self):
        from IPython.display import display
        display(self)


class LBRender:
    ctx: PIL.ImageDraw.ImageDraw
    img: PIL.Image.Image

    def __init__(self, ctx, mdp):
        self.ctx = ctx
        self.stroke_width = 0.5
        self.mdp = mdp

        self.set_projection(45)

        # DEFUNCT
        # # At present, we fix a 6x6 grid, and assume we want to center along y-axis for 0 height.
        # # In the future, we might want to compute this dynamically based on an MDP, so that it's perfectly centered.
        # # Might also want to incorporate max height.
        # self.set_offset_for_bounds(6, 6, 0)
        self.offset_y = self.ctx.height / 2
        self.offset_x = self.ctx.width / 2

    # def set_offset_for_bounds(self, x, y, height):
    #     # Temporary values to do this projection for offset
    #     self.offset_x = 0
    #     self.offset_y = 0

    #     # We compute the projection for a reasonable extent, given the bounds, given in unit cube space.
    #     extent_x, extent_y = self.project_from_unit_cube(x, height, y)
    #     # assert math.isclose(extent_x, 0) and extent_y < 0, (extent_x, extent_y)

    #     self.offset_y = self.ctx.height / 2 - extent_y / 2
    #     self.offset_x = self.ctx.width / 2

    def set_projection(self, horizontal_rotation, *, vertical_rotation=45, degrees=True):
        if degrees:
            horizontal_rotation *= math.pi / 180
            vertical_rotation *= math.pi / 180

        alpha = horizontal_rotation
        beta = vertical_rotation

        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)
        sin_beta = math.sin(beta)
        cos_beta = math.cos(beta)

        # Compute the projection matrix. Named to be consistent with http://en.wikipedia.org/wiki/Isometric_projection#Overview
        self.projection = [
            [cos_beta, 0, -sin_beta],
            [sin_alpha * sin_beta, cos_alpha, sin_alpha * cos_beta],
            [cos_alpha * sin_beta, -sin_alpha, cos_alpha * cos_beta],
        ]

    def _project(self, x, y, z):
        p = self.projection
        return (
            self.offset_x + (p[0][0] * x + p[0][1] * y + p[0][2] * z),
            self.offset_y - (p[1][0] * x + p[1][1] * y + p[1][2] * z)
        )
        return (
            self.offset_x + 0.707 * x - 0.707 * z,
            # self.ctx.height - (self.offset_y + 0.321 * x + 0.891 * y + 0.321 * z),
            self.offset_y - (0.321 * x + 0.891 * y + 0.321 * z),
        )

    def project_from_unit_cube(self, x, height, y):
        '''
        This projects from a space where cubes are axis-aligned and have unit side lenght into world space,
        where the height is half the size of the sides.
        '''
        edge_length = 35
        # Center x and y coordinates first.
        w, h = self.mdp.map_h.shape
        x -= w / 2
        y -= h / 2
        return self._project(x * edge_length, height * 0.5 * edge_length, y * edge_length)

    def draw_box(self, x, y, height, *, is_activated_light=None):
        self.ctx.polygon([
            self.project_from_unit_cube(x, 0, y),
            self.project_from_unit_cube(x + 1, 0, y),
            self.project_from_unit_cube(x + 1, height, y),
            self.project_from_unit_cube(x, height, y),
        ], COLORS.front, COLORS.stroke, self.stroke_width)
        self.ctx.polygon([
            self.project_from_unit_cube(x, 0, y),
            self.project_from_unit_cube(x, height, y),
            self.project_from_unit_cube(x, height, y + 1),
            self.project_from_unit_cube(x, 0, y + 1),
        ], COLORS.side, COLORS.stroke, self.stroke_width)
        top = (
            COLORS.top_light_on if is_activated_light is True else
            COLORS.top_light_off if is_activated_light is False else
            COLORS.top)
        self.ctx.polygon([
            self.project_from_unit_cube(x, height, y),
            self.project_from_unit_cube(x + 1, height, y),
            self.project_from_unit_cube(x + 1, height, y + 1),
            self.project_from_unit_cube(x, height, y + 1),
        ], top, COLORS.stroke, self.stroke_width)

    def draw(self, *, state_trajectory=None, program=None):
        mdp = self.mdp
        if program is not None:
            _, trajectory = envs.interpret_program_record_path(mdp, program)
            state_trajectory = [s for s, a in trajectory]

        state = state_trajectory[-1] if state_trajectory else mdp.initial_state()

        for x in range(mdp.map_h.shape[0])[::-1]:
            for y in range(mdp.map_h.shape[1])[::-1]:
                height = mdp.map_h[x, y]
                light_idx = mdp.map_light[x, y]
                self.draw_box(x, y, height, is_activated_light=(
                    None if light_idx == envs.NO_LIGHT else
                    state.map_lit[light_idx] == envs.CONST_MAP_LIT_TRUE
                ))

        if state_trajectory:
            self.draw_trajectory(state_trajectory)

        self.draw_bot(state)

    def draw_bot(self, state):
        i, j = state.position[0], state.position[1]
        direction = state.direction
        left = 0
        full_height = 100
        top_margin = 12
        sprite_height = full_height - top_margin
        width = 80
        top = direction * full_height + top_margin
        x, y = self.project_from_unit_cube(
            i,
            self.mdp.map_h[i, j],
            j,
        )
        self.ctx.alpha_composite(
            SPRITES.crop((left, top, left+width, top+sprite_height)),
            tuple(map(int, (x - width / 2, y - sprite_height))),
        )

    def draw_trajectory(self, trajectory):
        '''
        At present, we draw trajectories in a simple way. Every occupied state has an inset
        square drawn, and transitions between squares are connected in a simple way.
        '''
        rel_from_center = 0.2
        mdp = self.mdp

        def p(box, rel_offset):
            '''
            Projection helper that projects from center of the box, offset by some dx/dy
            '''
            (x, y), height = box
            return self.project_from_unit_cube(
                x + 0.5 + rel_offset['dx'],
                height,
                y + 0.5 + rel_offset['dy'],
            )

        def project_for_direction(box, direction, no_move_offset=0):
            '''
            Returns the two points for a line of width 2*rel_from_center that starts at the box.
            '''
            delta = envs.DIRECTIONS[direction]
            dx, dy = delta
            if dx == 0:
                # When dx=0, movement is in dy. So, the line's width is along dx.
                rel_offset_key, no_move_rel_offset_key = ['dx', 'dy']
            else:
                rel_offset_key, no_move_rel_offset_key = ['dy', 'dx']
            return [
                p(box, {rel_offset_key: -rel_from_center, no_move_rel_offset_key: no_move_offset}),
                p(box, {rel_offset_key: +rel_from_center, no_move_rel_offset_key: no_move_offset}),
            ]

        def pgon_for_box_center(box):
            '''
            Helper for square, center and inset for box.
            '''
            return [
                p(box, dict(dx=-rel_from_center, dy=-rel_from_center)),
                p(box, dict(dx=rel_from_center, dy=-rel_from_center)),
                p(box, dict(dx=rel_from_center, dy=rel_from_center)),
                p(box, dict(dx=-rel_from_center, dy=rel_from_center)),
            ]

        pgons = []
        prev = trajectory[0]
        prev_box = (prev.position, mdp.map_h[prev.position])
        pgons.append(pgon_for_box_center(prev_box))

        for curr in trajectory[1:]:
            curr_box = (curr.position, mdp.map_h[curr.position])
            if not (
                # We skip light actions
                prev.map_lit != curr.map_lit or
                # We also skip turns.
                prev.direction != curr.direction
            ):
                # Draw box for current state.
                pgons.append(pgon_for_box_center(curr_box))

                # Draw connecting segment from previous box to next one.
                pdx, pdy = envs.DIRECTIONS[prev.direction]
                dx, dy = envs.DIRECTIONS[curr.direction]
                p1, p2 = project_for_direction(prev_box, prev.direction, no_move_offset=(pdx+pdy)*rel_from_center)
                p4, p3 = project_for_direction(curr_box, curr.direction, no_move_offset=-(dx+dy)*rel_from_center)
                pgons.append([p1, p2, p3, p4])

            prev = curr
            prev_box = curr_box

        for pgon in pgons:
            self.ctx.polygon(pgon, COLORS.trajectory)


class Extent:
    '''
    This is a mixin/parent class that computes the extent of drawn items; handy for computing a final canvas size.
    Only used in matplotlib backend at the moment.
    '''
    def __init__(self):
        self.min = (math.inf, math.inf)
        self.max = (-math.inf, -math.inf)

    def _add_point(self, pt):
        self.min = tuple(min(m, math.floor(p)) for m, p in zip(self.min, pt))
        self.max = tuple(max(m, math.ceil(p)) for m, p in zip(self.max, pt))

    def polygon(self, pts, c, *args):
        for pt in pts:
            self._add_point(pt)

    def alpha_composite(self, im, dest):
        left, top = dest
        right = left + im.width
        bottom = top + im.height
        for x in [left, right]:
            for y in [top, bottom]:
                self._add_point((x, y))

class LBRMPL(Extent):
    '''
    A matplotlib shim for the subset of PIL that we use above. Doing this for vector rendering.
    '''

    dpi = 96

    def __init__(self, f, ax, w, h):
        super().__init__()
        self.f = f
        self.ax = ax
        self.width = w
        self.height = h

    def _c(self, c):
        '''
        Converts a uint8 color [0, 255] to an MPL color, floating point, [0, 1].
        '''
        return [c/255 for c in c]

    def polygon(self, pts, c, *args):
        super().polygon(pts, c, *args)
        kw = dict(linewidth=0)
        if args:
            stroke_color, stroke_width = args
            kw = dict(
                ec=self._c(stroke_color),
                linewidth=stroke_width,
            )
        a = mpl.patches.Polygon(
            pts,
            color=self._c(c),
            **kw,
        )
        self.ax.add_artist(a)

    def alpha_composite(self, im, dest):
        super().alpha_composite(im, dest)
        left, top = dest
        right = left + im.width
        bottom = top + im.height
        # HACK: zorder=10 won't always be appropriate...
        self.ax.imshow(im, extent=(left, right, bottom, top), zorder=10, interpolation='nearest')

    @classmethod
    def new(cls, *, size=DEFAULT_SIZE):
        w, h = size

        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html#figure-size-in-pixel
        px = 1/cls.dpi  # pixel in inches
        f, ax = plt.subplots(
            # figsize=(w*px, h*px),
            # xxx
            dpi=cls.dpi,
            # This is important for removing a frame that appears even when the axis is off.
            # https://gist.github.com/kylemcdonald/bedcc053db0e7843ef95c531957cb90f
            frameon=False,
        )
        # Turn axis off
        ax.axis('off')
        # Flip the y axis. By default, our bound is the whole image.
        # ax.set(ylim=[h, 0], xlim=[0, w])
        # xxx
        # Add a background
        ax.add_artist(mpl.patches.Rectangle((0, 0), w, h, color='w'))
        # For some strange reason, we need this for things to render better (i.e. fixes issues with margins) when displayed.
        f.set_facecolor('#ffffff')
        plt.tight_layout(pad=0)

        # Close to avoid showing figure in jupyter.
        plt.close(f)

        obj = cls(f, ax, w, h)
        obj.set_bounds(0, w, 0, h)
        return obj

    def set_bounds(self, left, right, top, bottom):
        assert left < right, (left, right)
        assert top < bottom, (top, bottom)
        w = right - left
        h = bottom - top

        # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_size_units.html#figure-size-in-pixel
        px = 1/self.dpi  # pixel in inches
        self.f.set_size_inches(w*px, h*px)

        # Flip the y axis. By default, our bound is the whole image.
        self.ax.set(xlim=[left, right], ylim=[bottom, top])

    def save(self, fn, *, tight=True):
        root, ext = os.path.splitext(fn)
        EXTS = ('.pdf', '.png')
        assert ext in EXTS, (root, ext)

        if tight:
            pad = 5
            self.set_bounds(
                self.min[0] - pad, self.max[0] + pad,
                self.min[1] - pad, self.max[1] + pad,
            )

        self.show()
        for ext in EXTS:
            # https://stackoverflow.com/questions/13714454/specifying-and-saving-a-figure-with-exact-size-in-pixels
            plt.savefig(
                f'{root}{ext}',
                facecolor=self.f.get_facecolor(),
                dpi=self.dpi,
            )
        plt.close(self.f)


    def show(self):
        plt.sca(self.ax)


def render(
        mdp, *,
        program=None,
        display=False,
        ctx_cls=LBRMPL,
        ctx_kw={},
        save_fn=None,
        save_kw={},
    ):
    ctx = ctx_cls.new(**ctx_kw)
    r = LBRender(ctx, mdp)
    r.draw(program=program)
    if save_fn is not None:
        r.ctx.save(save_fn, **save_kw)
    if display:
        r.ctx.show()
