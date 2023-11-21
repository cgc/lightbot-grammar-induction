import lb
from . import tools
import re

mkprog = tools.mkprog

def _gen_programs(trace, *, reorder_subroutines=False, assert_canonical_matches_reordered=False, **kwargs):
    cts = lb.all_substring_counts(trace, avoid_overlapping=True)
    rv = set()
    for p in lb.gen_programs(trace, cts, **kwargs):
        reordered = lb.ReorderSubroutineVisitor.reorder_subroutines(p)
        rv.add(reordered if reorder_subroutines else p)
        if assert_canonical_matches_reordered:
            canon = lb.canonicalize_program(p)
            assert reordered == canon
    return rv

def test_with_new_sr():
    for fn in [
        lb.with_new_sr,
        # Testing this old implementation since it's a simpler reference
        lb.with_new_sr_deprecated,
    ]:
        p = lb.Program('ABAB', ('ABAB',)*4)
        assert fn(p, 'ABAB') is p, 'reject case without space for another SR'

        p = lb.Program('ABA1', ('ABABA', '', '', ''))
        assert fn(p, 'A') is p, 'reject short subroutines'
        assert fn(p, 'ABABA') is p, 'we reject something that appears only once'
        assert fn(p, 'FGG') is p, 'we reject something that does not appear'
        assert fn(p, 'ABA') == lb.Program(main='21', subroutines=('2BA', 'ABA', '', ''))
        # Test suffix handling -- by default we don't ensure suffix is replaced
        assert fn(mkprog('ABAABABA'), 'ABA') == mkprog('11BA', 'ABA')

    # We ensure suffix is replace on right when this flag is passed
    assert lb.with_new_sr(mkprog('ABAABABA'), 'ABA', sr_is_suffix=True) == mkprog('1AB1', 'ABA')


def test_gen_programs():
    t = 'ABABA'
    counts = lb.all_substring_counts(t, avoid_overlapping=False)
    assert counts == {
        'BA': 2, 'AB': 2,
        'ABA': 2, 'BAB': 1,
        'ABAB': 1, 'BABA': 1,
    }
    assert _gen_programs(t) == {
        mkprog('ABABA'),
        mkprog('11A', 'AB'),
        mkprog('A11', 'BA'),
        # recursive
        mkprog('A1', 'BA1'),
        mkprog('1', 'AB1'),
        # repeating things that match endings
        mkprog('111', 'AB'),
        # NOTE: This is an example of a program that we used to generate but don't anymore, because the candidate
        # subroutine only appeared twice if overlaps were ok.
        # mkprog('1B1', 'ABA'),
    }

    t = 'ABCDABCEAB'
    counts = lb.all_substring_counts(t)
    assert {k for k, ct in counts.items() if ct >= 2} == {'AB', 'BC', 'ABC'}
    assert _gen_programs(t) == {
        mkprog('ABCDABCEAB'),
        mkprog('1CD1CE1', 'AB'),
        mkprog('A1DA1EAB', 'BC'),
        mkprog('1D1EAB', 'ABC'),
        mkprog('1D1E2', '2C', 'AB'),
        # repeating things that match endings
        mkprog('1D1E1', 'ABC'),
        mkprog('A1DA1EA1', 'BC'),
    }

    assert _gen_programs('ABCAB') == {
        mkprog('ABCAB'),
        mkprog('1C1', 'AB'),
    }

    expected = {
        mkprog('ABCABCAB'),
        mkprog('11AB', 'ABC'),
        mkprog('211', 'C2', 'AB'),
        mkprog('A11B', 'BCA'),
        mkprog('AB11', 'CAB'),
        mkprog('112', '2C', 'AB'),
        mkprog('1C1C1', 'AB'),
        mkprog('A1A1AB', 'BC'),
        mkprog('AB1B1B', 'CA'),

        # recursive
        mkprog('1', 'ABC1'),
        mkprog('A1', 'BCA1'),
        mkprog('AB1', 'CAB1'),
        mkprog('21', 'C21', 'AB'),

        # extensions
        mkprog('111', 'ABC'),
        mkprog('A1A1A1', 'BC'),
        mkprog('A112', '2A', 'BC'),
        mkprog('A111', 'BCA'),
    }

    assert _gen_programs('ABCABCAB') == expected

    assert _gen_programs(mkprog('WWWSLJLWWWSRJRWWWS').main) == {
        mkprog('1LJL1RJR1', 'WWWS'),
        mkprog('1SLJL1SRJR1S', 'WWW'),
        mkprog('W1LJLW1RJRW1', 'WWS'),
        mkprog('WW1LJLWW1RJRWW1', 'WS'),
        mkprog('1WSLJL1WSRJR1WS', 'WW'),
        mkprog('21LJL21RJR21', 'WS', 'WW'),
        mkprog('WWWSLJLWWWSRJRWWWS'),
    }

    ps = _gen_programs('ABCDABCEBCDBC')
    # HACK A quirk in our program enumeration is that overlapping subroutines have
    # an order of replacement.
    assert mkprog('1CD1CE2D2', 'AB', 'BC') in ps

    # Some recursion checks
    ps = _gen_programs('ABABABAB')
    assert mkprog('1', 'AB1') in ps
    assert mkprog('1', 'ABAB1') in ps
    assert mkprog('AB1', 'AB1') not in ps

    # This is a weird issue when there's a potential suffix for extending the trace.
    # Because string replace is usually left-to-right, the parse is ambiguous, but
    # then doens't make sense because there are extra primitive instructions that aren't
    # in a subroutine.
    ps = _gen_programs('CADCCADC')
    assert mkprog('11', 'CADC') in ps
    assert mkprog('1CAD1', 'CADC') in ps
    assert mkprog('11ADC', 'CADC') not in ps

    # This tests an issue where overlapping substrings can result in multiple programs
    t = 'ABABABAB'
    progs = list(lb.gen_programs(t, lb.all_substring_counts(t, avoid_overlapping=False)))
    import collections
    assert len(progs) == len(set(progs)), [
        (ct_, p) for p, ct_ in collections.Counter(progs).items() if ct_ > 1]

    # A combination of recursion & overshooting goal
    ps = _gen_programs('ABCD'*2+'AB')
    assert mkprog('1', 'ABCD1') in ps

    # Some weird tests of repetition + recursion
    ps = _gen_programs('ABABCABABC')
    assert mkprog('1', '22C1', 'AB') in ps

    # A weird case where the recursive program is repeated as a prefix
    ps = _gen_programs('ABCABCDABCAB')
    assert mkprog('22D1', '21', 'ABC') in ps

    # Adding an explicit test for the limits of our extension-based setup.
    # At present, we avoid generating trace extensions if the substring isn't
    # separately present the minimum number of times.
    ps = _gen_programs('ABCDEABCDE')
    assert mkprog('11', 'ABCDE') in ps
    assert mkprog('A11', 'BCDEA') not in ps

    # This is a nice test for nested subroutines with trace extension; all 3 subroutines ABCD, BCD, CD are nested
    # and appear twice, however the program is refactored. So, this is a good test to make sure it works correctly.
    ps = _gen_programs('CDABCDABCDBC')
    assert mkprog('CD11BC', 'ABCD') in ps
    assert mkprog('CDA1A11', 'BCD') in ps
    assert mkprog('1AB1AB1B1', 'CD') in ps
    assert mkprog('CD112', 'A2', 'BCD') in ps
    assert mkprog('211B2', 'AB2', 'CD') in ps
    assert mkprog('2A1A11', 'B2', 'CD') in ps
    assert mkprog('3112', 'A2', 'B3', 'CD') in ps


def test_trace():
    base_mdp = lb.EnvLoader.maps[0]
    mdp = lb.LightbotTrace(base_mdp)
    s = mdp.initial_state()
    assert s.light_target is None
    s = mdp.next_state(s, (3, 2))
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'C')
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'C')
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'C')
    assert s.light_target == (3, 2)
    s = mdp.next_state(s, 'A')
    assert s.light_target is None
    assert mdp.is_terminal(s)
    assert s.trace == 'CCCA'

def test_all_substring_counts():
    len2 = {'AB': 2, 'BA': 2, 'ABA': 2, 'BAB': 1, 'ABAB': 1, 'BABA': 1}
    assert lb.all_substring_counts('ABABA', min_length=2, avoid_overlapping=False) == len2
    assert lb.all_substring_counts('ABABA', min_length=1, avoid_overlapping=False) == len2 | {'A': 3, 'B': 2}
    assert lb.all_substring_counts('ABABA', min_length=2, avoid_overlapping=True) == len2 | {'ABA': 1}

    def _greater_count_than_1(d):
        return {k: v for k, v in d.items() if v > 1}
    exp = {
        'AB': 3,
        'BC': 2,
        'CA': 2,
        'ABC': 2,
        'BCA': 2,
        'CAB': 2,
    }
    overlaps = {
        'ABCA': 2,
        'ABCAB': 2,
        'BCAB': 2,
    }
    assert _greater_count_than_1(
        lb.all_substring_counts('ABCABCAB', min_length=2, avoid_overlapping=False)) == exp | overlaps
    assert _greater_count_than_1(
        lb.all_substring_counts('ABCABCAB', min_length=2, avoid_overlapping=True)) == exp
