import functools
from . import envs

I = envs.INSTRUCTIONS

readable_to_inst = {
    "W": I.WALK,
    "J": I.JUMP,
    "R": I.RIGHT,
    "L": I.LEFT,
    "S": I.LIGHT,
}

def mkinst(*args):
    rv = []

    for a in ''.join(args):
        if a in (' ', ','):
            continue
        assert a in readable_to_inst or a in I
        rv.append(readable_to_inst.get(a, a))
    return ''.join(rv)

def mkprog(main, *srs):
    return envs.Program(
        main=mkinst(main),
        subroutines=tuple(mkinst(srs[i]) if i < len(srs) else '' for i in range(4)),
    )

def readable_repr(p):
    def readable_inst(seq):
        for readable, inst in readable_to_inst.items():
            seq = seq.replace(inst, readable)
        return seq
    if not isinstance(p, envs.Program):
        return readable_inst(p)
    return envs.Program(
        main=readable_inst(p.main),
        subroutines=tuple(
            readable_inst(sr)
            for sr in p.subroutines
        ),
    )

def method_cache(fn):
    '''
    Avoiding functools.lru_cache since it holds onto object references
    when applied to methods of an object, creating a referenc cycle (not great for GC).
    Instead, preferring a strategy where cache storage lives on the object.
    '''
    cache_attr = '_cache_'+fn.__name__
    cache_info_attr = '_cache_info_'+fn.__name__
    @functools.wraps(fn)
    def wrapped(self, *args, **kwargs):
        # Since the store is object-local, we always have to ensure
        # objects passed in have the cache initialized.
        if not hasattr(self, cache_attr):
            setattr(self, cache_attr, {})
            setattr(self, cache_info_attr, dict(hits=0, misses=0))
        cache = getattr(self, cache_attr)
        cache_info = getattr(self, cache_info_attr)

        # Now we check for this function call, and
        # run the function if it hasn't been called before.
        key = (args, frozenset(kwargs.items()) if kwargs else None)
        if key not in cache:
            cache[key] = fn(self, *args, **kwargs)
            cache_info['misses'] += 1
        cache_info['hits'] += 1
        return cache[key]
    return wrapped
