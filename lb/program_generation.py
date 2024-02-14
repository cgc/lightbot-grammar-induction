from . import envs as main
from . import program_analysis
from collections import namedtuple
import functools
import re
import warnings


def with_new_sr_deprecated(p, sr):
    warnings.warn('This old/simpler SR function is deprecated')
    if len(sr) == 1:
        return p

    for idx in range(len(main.PROCESS_INST)):
        if not p.subroutines[idx]:
            inst = main.PROCESS_INST[idx]
            break
    else:
        return p

    if sr not in p.main and all(sr not in sr_ for sr_ in p.subroutines):
        return p

    m = p.main.replace(sr, inst)
    srs = [sr_.replace(sr, inst) for sr_ in p.subroutines]

    initial_len = len(p.main) + sum(len(sr_) for sr_ in p.subroutines)
    post_rewrite_len = len(m) + sum(len(sr_) for sr_ in srs)

    len_if_two_sites_were_rewritten = initial_len - 2 * (len(sr) - 1)

    if post_rewrite_len > len_if_two_sites_were_rewritten:
        return p
    srs[idx] = sr

    return main.Program(
        main=m,
        subroutines=tuple(srs)
    )


def right_replace(s, old, new, replacement_count):
    # https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
    return new.join(s.rsplit(old, replacement_count))


def gen_single_site_drops(str, rx):
    '''
    This routine returns all versions of the string
    generated by dropping at most one match of the supplied
    regular expression.

    >>> list(gen_single_site_drops('abaca', re.compile('a')))
    ['baca', 'abca', 'abac']
    >>> list(gen_single_site_drops('abaca', re.compile('a|b')))
    ['baca', 'aaca', 'abca', 'abac']
    >>> list(gen_single_site_drops('ab1ab2ab', re.compile('ab')))
    ['1ab2ab', 'ab12ab', 'ab1ab2']
    '''
    for m in rx.finditer(str):
        yield str[:m.start()] + str[m.end():]

def with_new_sr(p, sr, *, sr_is_suffix=False):
    '''
    Attempts to rewrite a program by making the action sequence `sr` into a new subroutine.

    When the program cannot be rewritten, the supplied program is returned to make it simple
    to test if the program was rewritten without requiring an equality test.
    '''
    if len(sr) == 1:
        return p

    # Identify the next SR location.
    for idx in range(len(main.PROCESS_INST)):
        if not p.subroutines[idx]:
            inst = main.PROCESS_INST[idx]
            break
    else:
        return p

    # If the candidate SR doesn't occur in any part of the program, then skip it.
    for sr_ in p.subroutines:
        if sr in sr_:
            in_any = True
            break
    else:
        in_any = False
    if sr not in p.main and not in_any:
        return p

    m = p.main
    if sr_is_suffix:
        # If the SR is meant to be a suffix (like in recursion), then we make sure to perform a replacement on the right.
        assert m.endswith(sr)
        m = right_replace(m, sr, inst, 1)

    # Replace all occurences of the SR with the new instruction.
    m = m.replace(sr, inst)
    srs = [sr_.replace(sr, inst) for sr_ in p.subroutines]

    initial_len = len(p.main)
    for sr_ in p.subroutines: initial_len += len(sr_)

    post_rewrite_len = len(m)
    for sr_ in srs: post_rewrite_len += len(sr_)

    len_if_two_sites_were_rewritten = initial_len - 2 * (len(sr) - 1)

    # This is a simple way to test whether sufficient sites were rewritten
    if post_rewrite_len > len_if_two_sites_were_rewritten:
        return p
    srs[idx] = sr

    return main.Program(
        main=m,
        subroutines=tuple(srs)
    )


def gen_programs(
    trace, counts, *,
    min_repetitions=2,
):
    # Sort to replace longest strings first
    subprogs = sorted([
        subprog
        for subprog, ct in counts.items()
        if ct >= min_repetitions
    ], key=lambda s: (-len(s), s))

    def _recur(p, *, i=0, trace_extension_length=False):
        '''
        This recursive function implements a simple backtracking scheme for enumerating
        all possible subsets of the subroutines, in order from longest to shortest.
        '''
        recur_kw = dict(i=i+1, trace_extension_length=trace_extension_length)
        # trace_extension_length measures how much the trace length was extended by. We use this
        # to efficiently bail out if we extended the trace but are now only considering short subroutines.
        missing_extension = trace_extension_length and p.main[-1] not in main.PROCESS_INST

        if i == len(subprogs):
            if missing_extension:
                # If we extended the trace, but do not end with a subroutine call, then we should reject this program.
                return
            yield p
            return

        sr = subprogs[i]

        # This line checks that we haven't missed the chance to append actions to the
        # trace that are of sufficient length. It assumes that SRs are considered
        # in descending length, from long to short.
        if missing_extension and len(sr) <= trace_extension_length:
            return

        # First, we continue to the next one, without using this SR to rewrite
        yield from _recur(p, **recur_kw)

        if trace_extension_length and p.main.endswith(sr):
            # When we need to replace a suffix, we rewrite the program to ensure
            # the suffix is replaced, by using right_replace
            p_with_subprog = with_new_sr(p, sr, sr_is_suffix=True)
        else:
            p_with_subprog = with_new_sr(p, sr)

        # If the program could be rewritten, then we recurse to consider other subroutines.
        if p != p_with_subprog:
            yield from _recur(p_with_subprog, **recur_kw)

    def try_to_make_tail_recursive(trace, sr):
        t = trace

        # Now we actually try to remove the repeated suffix/subroutine and count them
        repetitions = 0
        while t.endswith(sr):
            repetitions += 1
            t = t[:-len(sr)]

        # This filters out cases that don't have enough repetition
        if repetitions >= min_repetitions:
            inst = main.INSTRUCTIONS.PROCESS1
            p = main.Program(main=t+inst, subroutines=(sr+inst, '', '', ''))
            yield from _recur(p)

    def newprog(trace):
        return main.Program(main=trace, subroutines=('',)*4)

    # Try all tail-recursive programs that give us this trace by checking all suffixes
    for i in range(1, len(trace)):
        suffix = trace[-i:]
        # First, a check that will leave some false positives (since repetitions might not be at end, like ABXAB with sr=AB)
        # note: counts can be 0 if suffix is len=1
        # This is actually checked in try_to_make_tail_recursive(.)
        if counts.get(suffix, 0) >= min_repetitions:
            yield from try_to_make_tail_recursive(trace, suffix)

    # Consider all trace extensions that could come from use of a subroutine at the end of the program
    # These have post-goal actions.
    extended_traces = {}
    for sr in subprogs:
        for i in range(1, len(sr)):
            prefix, rest = sr[:i], sr[i:]
            assert len(prefix) >= 1 and len(rest) >= 1

            if trace.endswith(prefix):
                extended_traces.setdefault(trace + rest, []).append(sr)

    # Generate programs for traces with post-goal actions
    for extended_trace, srs in extended_traces.items():
        yield from _recur(newprog(extended_trace), trace_extension_length=len(extended_trace) - len(trace))
        for sr in srs:
            # As a special case, we also try to see if this suffix makes a good recursive program
            yield from try_to_make_tail_recursive(extended_trace, sr)

    # This is our standard case for generating programs
    yield from _recur(newprog(trace))



def all_substring_counts(t, *, min_length=2, avoid_overlapping=True):
    '''
    This function counts all substrings greater than some minimum length,
    returned in a dictionary from substring to count. These are used as candidate
    subroutines in program generation.
    '''
    ct = {}
    if avoid_overlapping:
        last = {}
    for length in range(min_length, len(t)):
        for start in range(len(t) - length + 1):
            st = t[start:start+length]
            if (
                avoid_overlapping and
                st in last and
                # We want the last appearance to have ended at or before this index.
                not (last[st] + length <= start)
            ):
                continue
            if st not in ct:
                ct[st] = 0
            ct[st] += 1
            if avoid_overlapping:
                last[st] = start
    return ct
