# Several loops are flattened into one before extraction

A function with several loops, or with a jump into a loop past its head, is
rewritten at the GOTO level into a single loop over a location variable,
before the k-induction transformation, when the transition-system strategy is
requested and main has at least two cut points: two loop heads, or a loop
head and a jump that enters past it. Extraction,
slicing, BTOR2 export and the engines stay single-loop. We rejected a
program-counter encoding inside extraction (one set of markers per loop head):
it builds the same transition system with more code in the most fragile file.
Programs that are extractable without flattening take exactly the same path as
before, so they pay nothing for it.

## Consequences

- When extraction still fails, the fallback BMC runs on the flattened program:
  an explicit `--unwind N` counts dispatcher steps, not iterations of each
  original loop.
- Interval analysis sees one loop head, so its bounds join every cut point.
- The location is an unsigned bit-vector just wide enough for the cut points.
