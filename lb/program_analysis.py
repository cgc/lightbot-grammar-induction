import pandas as pd
from . import envs
import collections

def replace_until_unchanged(seq, reps, *, limit=1000):
    for _ in range(limit):
        prev = seq
        for s, rep in reps.items():
            seq = seq.replace(s, rep)
        if prev == seq:
            break
    return seq

def light_order(seq):
    return replace_until_unchanged(seq, {
        envs.INSTRUCTIONS.LEFT+envs.INSTRUCTIONS.LIGHT: envs.INSTRUCTIONS.LIGHT+envs.INSTRUCTIONS.LEFT,
        envs.INSTRUCTIONS.RIGHT+envs.INSTRUCTIONS.LIGHT: envs.INSTRUCTIONS.LIGHT+envs.INSTRUCTIONS.RIGHT,
    })


def replace_noop_sequences(seq):
    return replace_until_unchanged(seq, envs.NO_OP_MAP)


def inline_subroutine_in_instructions(seq, instr, subroutine):
    if instr not in seq:
        return seq
    rv = ''
    for i in seq:
        if i == instr:
            rv += subroutine
        else:
            rv += i
    return rv


def inline_subroutine_in_program(program, instruction):
    srs = program.subroutines
    sr = srs[envs.PROCESS_TO_IDX[instruction]]
    assert instruction not in sr, 'cannot inline recursive routine'
    return envs.Program(
        main=inline_subroutine_in_instructions(program.main, instruction, sr),
        subroutines=tuple(
            # Delete the subroutine if inlining.
            '' if instruction == p else
            # Otherwise, inline it.
            inline_subroutine_in_instructions(srs[idx], instruction, sr)
            for idx, p in enumerate(envs.IDX_TO_PROCESS)
        ),
    )



def inline_single_ref_and_dead_code(prog):
    # Inline routines called once, and remove "dead code", unused subroutines.
    # Could do more with dead code detection; recursion makes it possible to write noops after the recursive call

    recursion = False

    refcounts = collections.Counter()
    visited_procs = set()
    def visit(instructions, stack):
        if stack:
            # We are executing a process if we have a stack entry; the current process is the last stack entry.
            curr_proc = stack[-1]
            if curr_proc in visited_procs:
                return
            visited_procs.add(curr_proc)
        for instruction in instructions:
            if instruction not in envs.PROCESS_INST:
                continue
            refcounts[instruction] += 1
            if instruction in stack:
                recursion = True
                return
            visit(prog.subroutines[envs.PROCESS_TO_IDX[instruction]], stack + [instruction])

    visit(prog.main, [])

    for instruction in envs.PROCESS_INST:
        if refcounts[instruction] == 0:
            prog = envs.Program(
                main=prog.main,
                subroutines=tuple(
                    # Delete the subroutine
                    '' if instruction == p else prog.subroutines[idx]
                    for idx, p in enumerate(envs.IDX_TO_PROCESS)
                ),
            )
        if refcounts[instruction] == 1:
            prog = inline_subroutine_in_program(prog, instruction)

    return prog


def inline_len1_subroutine(program):
    for instruction in envs.PROCESS_INST:
        if len(program.subroutines[envs.PROCESS_TO_IDX[instruction]]) == 1:
            program = inline_subroutine_in_program(program, instruction)
    return program



def without_dead_and_noop_instructions(mdp, p):
    live = envs.Program(
        [False] * len(p.main),
        tuple(
            [False] * len(sr)
            for sr in p.subroutines
        ),
    )

    stack = []

    def recur(s, name):
        instructions = p.main if name is None else p.subroutines[envs.PROCESS_TO_IDX[name]]
        curr_live = live.main if name is None else live.subroutines[envs.PROCESS_TO_IDX[name]]
        for idx, inst in enumerate(instructions):
            if inst in envs.PROCESS_INST:
                if inst in stack and mdp.is_terminal(s):
                    raise RecursionDetected()
                if not mdp.is_terminal(s):
                    curr_live[idx] = True # If we moved this line before the recursion error, we would keep single-use recursive subroutines
                stack.append(inst)
                s = recur(s, inst)
                assert stack.pop() == inst
            else:
                ns = mdp.next_state(s, inst)
                if s != ns and not mdp.is_terminal(s):
                    curr_live[idx] = True
                s = ns
        return s
    try:
        recur(mdp.initial_state(), None)
    except RecursionDetected:
        pass

    def live_filter(instructions, live):
        return ''.join([i for i, l in zip(instructions, live) if l])

    return envs.Program(
        live_filter(p.main, live.main),
        tuple(
            live_filter(sr, lsr)
            for sr, lsr in zip(p.subroutines, live.subroutines)
        ),
    )




def canonicalize_program(p, mdp=None):
    # We first maximize subroutine use.
    # I think this makes sense, since it maximizes SR use given what people wrote.
    # However, single refs + no-ops will expose possible SR use sites, so this
    # function does not return a canonical program that is a fixed point.
    # TODO: Consider making this avoid SRs with 1 use?
    p = ensure_max_subroutine_use(p)

    if mdp is not None:
        p = without_dead_and_noop_instructions(mdp, p)
    else:
        import warnings
        warnings.warn("Pass MDP to canonicalize_program(p, mdp) for processing")

    # Inline routines called once, and remove "dead code", unused subroutines.
    p = inline_single_ref_and_dead_code(p)

    # Handle subroutines containing only one instruction
    p = inline_len1_subroutine(p)

    # Handles no-op sequences.
    p = envs.Program(
        main=replace_noop_sequences(p.main),
        subroutines=[replace_noop_sequences(sr) for sr in p.subroutines],
    )

    p = ReorderSubroutineVisitor.reorder_subroutines(p)

    return p


def canonicalized_trace(mdp, program):
    _, path = envs.interpret_program_record_path(mdp, program, state=mdp.initial_state())

    # Asserting that the last element has no instruction
    assert path[-1][-1] is None

    rv = []
    for pair in path:
        state, instruction = pair
        if instruction is None:
            rv.append(pair)
        else:
            next_state = mdp.next_state(state, instruction)
            if state != next_state:
                rv.append(pair)

    t = ''.join(inst for state, inst in rv if inst is not None)
    t = replace_noop_sequences(t)
    # Sort of a hack: dropping instructions that are now post-terminal b/c of light reordering.
    # One example is WWWLJS on maps[0]
    if mdp.is_terminal(path[-1][0]):
        t = t[:t.rindex(envs.INSTRUCTIONS.LIGHT)+1]
    return t


class ProgramVisitor(object):
    def visit_primitive(self, program, instruction):
        pass
    def visit_process(self, program, instruction):
        sr = program.main if instruction is None else program.subroutines[envs.PROCESS_TO_IDX[instruction]]
        for instruction in sr:
            self.visit(program, instruction)
    def visit(self, program, instruction=None):
        # instruction is None for main
        if instruction is None or instruction in envs.PROCESS_INST:
            self.visit_process(program, instruction)
        else:
            self.visit_primitive(program, instruction)

class RecursionDetected(Exception):
    pass

class DetectRecursionProgramVisitor(ProgramVisitor):
    def __init__(self):
        super().__init__()
        self.stack = []
    def visit_process(self, program, instruction):
        if instruction in self.stack:
            raise RecursionDetected()
        self.stack.append(instruction)
        try:
            super().visit_process(program, instruction)
        finally:
            self.stack.pop()
    @classmethod
    def is_recursive(cls, program):
        try:
            cls().visit(program)
            return False
        except RecursionDetected:
            return True


class ReorderSubroutineVisitor(DetectRecursionProgramVisitor):
    def __init__(self):
        super().__init__()
        self.mapping = {}
    def visit_process(self, program, instruction):
        next_idx = len(self.mapping)
        if instruction is not None and instruction not in self.mapping:
            self.mapping[instruction] = envs.IDX_TO_PROCESS[next_idx]
        super().visit_process(program, instruction)

    @classmethod
    def reorder_subroutines(cls, program):
        inst = cls()
        try:
            inst.visit(program)
        except RecursionDetected:
            pass

        return rename_subroutines(program, inst.mapping)


def rename_subroutines(program, mapping):
    def swap(instrs):
        return ''.join(mapping[i] if i in envs.PROCESS_TO_IDX else i for i in instrs)

    srs = [''] * len(envs.PROCESS_INST)
    for source, dest in mapping.items():
        srs[envs.PROCESS_TO_IDX[dest]] = swap(program.subroutines[envs.PROCESS_TO_IDX[source]])
    return envs.Program(main=swap(program.main), subroutines=tuple(srs))


class FlattenProgram(DetectRecursionProgramVisitor):
    def __init__(self):
        super().__init__()
        self.flat = ''
    def visit_primitive(self, program, instruction):
        self.flat += instruction

    @classmethod
    def flatten(cls, program):
        inst = cls()
        try:
            inst.visit(program)
        except RecursionDetected:
            pass

        return inst.flat


class ChildFirstSort(DetectRecursionProgramVisitor):
    def __init__(self):
        super().__init__()
        self.order = []

    def visit_process(self, program, instruction):
        try:
            super().visit_process(program, instruction)
        except RecursionDetected:
            pass
        if instruction is not None and instruction not in self.order:
            self.order.append(instruction)

    @classmethod
    def sorted(cls, program):
        obj = cls()
        try:
            obj.visit(program)
        except RecursionDetected:
            pass
        for inst, sr in zip(envs.PROCESS_INST, program.subroutines):
            if sr and inst not in obj.order:
                obj.visit(program, instruction=inst)
        return obj.order


def ensure_max_subroutine_use(p):
    order = ChildFirstSort.sorted(p)
    for inst in order:
        idx = envs.PROCESS_TO_IDX[inst]
        sr = p.subroutines[idx]
        if not sr:
            continue
        p = envs.Program(
            main=p.main.replace(sr, inst),
            subroutines=tuple(
                # No replacement for current SR
                sr_ if idx == idx_ else
                # Replace otherwise
                sr_.replace(sr, inst)
                for idx_, sr_ in enumerate(p.subroutines)
            )
        )
    return p


ProgramCounts = collections.namedtuple('ProgramCounts', ['step', 'step_noop', 'step_post', 'step_post_noop'])
class ProgramStepCounter(ProgramVisitor):
    def __init__(self, mdp):
        super().__init__()
        self.stack = []
        self.mdp = mdp
        self.state = mdp.initial_state()
        self.step = 0
        self.step_noop = 0
        self.step_post = 0
        self.step_post_noop = 0

    def visit_primitive(self, program, instruction):
        ns = self.mdp.next_state(self.state, instruction)
        is_noop = self.state == ns
        if self.mdp.is_terminal(self.state):
            if is_noop:
                self.step_post_noop += 1
            else:
                self.step_post += 1
        else:
            if is_noop:
                self.step_noop += 1
            else:
                self.step += 1
        # Order matters: has to be after is_terminal() and self.state == ns
        self.state = ns

    def visit_process(self, program, instruction):
        if instruction in self.stack and self.mdp.is_terminal(self.state):
            return
        self.stack.append(instruction)
        super().visit_process(program, instruction)
        assert self.stack.pop() == instruction

    @classmethod
    def count(cls, mdp, program):
        inst = cls(mdp)
        inst.visit(program)
        return ProgramCounts(inst.step, inst.step_noop, inst.step_post, inst.step_post_noop)

class ProcessStateTracker(ProgramStepCounter):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.proc_to_state = {}
    def visit_process(self, program, instruction):
        if instruction is not None:
            self.proc_to_state.setdefault(instruction, []).append(self.state)
        return super().visit_process(program, instruction)
