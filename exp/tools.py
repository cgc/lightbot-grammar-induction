def compress_runs(seq, *, as_string=False):
    '''
    >>> assert compress_runs([0, 1, 1, 1, 3], as_string=True) == "['0', '1x3', '3']"
    >>> assert compress_runs([0, 0, 1, 1, 1, 3], as_string=True) == "['0x2', '1x3', '3']"
    '''
    prev = seq[0]
    res = [[prev, 1]]
    for x in seq[1:]:
        if x == prev:
            res[-1][-1] += 1
        else:
            res.append([x, 1])
        prev = x
    if as_string:
        s = []
        for x, ct in res:
            if ct == 1:
                s.append(f'{x}')
            else:
                s.append(f'{x}x{ct}')
        return str(s)
    return res
