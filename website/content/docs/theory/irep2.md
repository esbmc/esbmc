---
title: "IRep2: ESBMC's Internal Representation"
weight: 20
---

`irep2` is ESBMC's typed, reference-counted, copy-on-write internal
representation for **expressions** (`expr2t`) and **types** (`type2t`). It is
the data structure every frontend lowers to, every transformation rewrites,
and every backend (symex, SMT, goto2c) consumes. It replaces the older
"stringy" `irept` for the verification pipeline; conversions live in
`util/migrate.{h,cpp}`.

For the full documentation — design rationale, file layout, anatomy of a node,
how to add a new node, gotchas, and a reference of every type and expression
kind — see the README maintained alongside the source:

[**`src/irep2/README.md`** on GitHub](https://github.com/esbmc/esbmc/blob/master/src/irep2/README.md)

The source itself lives at [`src/irep2/`](https://github.com/esbmc/esbmc/tree/master/src/irep2).
