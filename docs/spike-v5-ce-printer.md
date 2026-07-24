# Scope — V.5 / W4: the IREP2-native counterexample printer

**Program:** Part V of `docs/irep2-migration.md` (IREP2-native frontend→goto, #4715).
**Question this scopes:** should the counterexample expression printer
(`c_expr2string` / `cpp_expr2string`, reached via `from_expr`) be migrated to
consume `expr2tc` natively, and if so, how is it sliced?
**Status (2026-07-23):** scoped; **provisional recommendation: DEFER** — the
migration is all-or-nothing in value and strict-byte-identity in cost, so it is a
poor fit for incremental work and low priority relative to the adjuster flip.
**Refs:** #4715; sibling keystones `docs/spike-v1k-w1loc.md` (W1-loc, native
goto-convert — largely drained) and `docs/scope-v1k-adjuster.md` (the
`python_adjust` flip).

---

## 1. What this is

The counterexample / trace printer turns an expression into a C- (or C++-)syntax
string for display in `--show-vcc`, the trace dump (`goto_trace.cpp`), and the
JSON/witness output (`goto-symex/json.cpp`). The worker is `c_expr2stringt`
(`src/util/c_expr2string.cpp`, ~2800 lines) and its C++ subclass
`cpp_expr2stringt`; the entry point is `from_expr` (`src/langapi/language_util.*`).

The trace itself is already IREP2: `goto_trace_stept` holds `expr2tc` members
(`lhs`, `rhs`, `value`, `original_lhs`, `output_args` —
`src/goto-symex/goto_trace.h:76-86`), and the callers pass those `expr2tc`
straight to `from_expr`. The overload that receives them
(`language_util.h:18-24`) does:

```cpp
inline std::string from_expr(const namespacet &ns, const irep_idt &id,
                             const expr2tc &expr, ...) {
  return from_expr(ns, id, migrate_expr_back(expr), target);  // <-- the back-hop
}
```

So every printed trace value round-trips IREP2 → legacy `exprt` and is then
walked by `c_expr2stringt::convert_rec`'s ~159-case dispatch on `exprt.id()`.
"V.5-native" means: dispatch `convert_rec` on `expr2tc` directly and delete that
`migrate_expr_back`.

## 2. Why it is a poor incremental target

**(a) The value is all-or-nothing.** The only thing the migration buys is
removing the `migrate_expr_back` at print time (a cleanliness / "no legacy at the
seam" win — the printer is not a hot path; it runs once per reported
counterexample). That back-hop cannot be removed until **every** kind
`convert_rec` can receive is handled natively, because any single fall-through to
the legacy path re-introduces it. Migrating 10 or 100 of the ~159 kinds removes
**zero** back-hops — the fallback still needs the full `exprt`. There is no
partial-credit; a half-migrated printer is strictly worse than either endpoint
(two dispatchers, one of which still round-trips).

**(b) The gate is strict per-kind byte-identity.** Counterexample text is
asserted verbatim by `test.desc` regexes across the suite. A native
`convert_rec` must reproduce `c_expr2stringt`'s exact output for every kind —
operator precedence and parenthesisation (`convert_rec`'s `precedence`
threading), constant formatting (base, suffix, sign, char/bool/float special
cases), the symbol **shorthand** machinery (`id_shorthand` / `get_shorthands` /
`get_symbols`, which disambiguates names against the whole expression), struct /
union / array / pointer rendering, and the `cpp_expr2stringt` overrides. Each
kind is a fiddly exact-match, and the shorthand logic is expression-global (not
per-node), so even "leaf" kinds like a bare symbol are not trivially local.

**(c) It duplicates a large surface.** ~159 cases across two classes
(`c_`/`cpp_`) would be reimplemented on `expr2tc`, doubling the printer until the
legacy one can be deleted — and it can only be deleted once **all** frontends'
`from_expr` users are on the native path and every kind is covered.

## 3. If it is done anyway — the only sound shape

Introduce `expr2string(ns, id, expr2tc)` that dispatches natively and falls back
to `migrate_expr_back` + the legacy printer for unmigrated kinds, gated on
**byte-identical output vs the legacy path** over the whole regression suite
(the counterexample regexes are the oracle). Migrate kinds in dependency order
(constants → symbols-with-shorthand → unary/binary-with-precedence → aggregates),
keeping the fallback until the census of reachable kinds is exhausted, then flip
the callers and delete the back-hop. **Do not** land partial migration as if it
were progress — track it against the all-or-nothing bar in §2(a).

## 4. Recommendation

**Defer.** Relative to the two live Part V efforts — the `python_adjust` flip
(`scope-v1k-adjuster.md`; replaces `clang_cpp_adjust` on the Python path, real
semantic payoff) and the W1-loc native goto-convert (`spike-v1k-w1loc.md`; the
native-kind ladder is drained, remaining fallbacks are marginal) — the printer
migration is the lowest-value Part V item: a non-hot-path cleanliness win with an
all-or-nothing payoff and a large strict-byte-identity surface. It should be
picked up only as a deliberate, single-owner sweep (not sliced across a loop),
and only after the higher-payoff flip work is landed or explicitly parked.
