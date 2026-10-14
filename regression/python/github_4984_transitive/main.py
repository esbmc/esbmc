from modstub import C

# Only C is named, but C.check -> base() -> TAG. The emitter's transitive
# closure must carry both the sibling helper `base` and the global `TAG`.
obj: C = C(7)
obj.check()
