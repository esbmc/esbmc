#pragma once

#include <util/context.h>
#include <util/namespace.h>
#include <irep2/irep2.h>
#include <string>
#include <vector>

/// V.1k (b) IREP2-native Python adjuster (phases B.0–B.2).
///
/// Replaces, for the Python frontend, the legacy `clang_cpp_adjust` round-trip
/// on Python output (`python_language.cpp`). It walks each code symbol's IREP2
/// value (`symbolt::get_value2()`) and follows a transient `symbol_type2t`
/// `member2t`/`index2t` source to its resolved `struct_type2t`/`array_type2t`
/// (the V.1k "two-phase source invariant": relaxed at construction, re-enforced
/// here before symex) — covering both a plain instance source and a
/// dereferenced instance pointer.
///
/// Wired into `python_languaget::typecheck` behind `--python-irep2-adjust`
/// (default off ⇒ byte-identical): it runs after `clang_cpp_adjust` and, until
/// the converter emits transient sources pre-adjust, resolves nothing — so the
/// path is dead-but-tested, mirroring the "add the machinery, prove it inert,
/// wire it later" pattern (esbmc/esbmc#5265). `#cpp_type`/`#member_name`
/// carriage and dropping the legacy hop remain later phases (B.4/B.5). See
/// `docs/irep2-migration.md`, section "V.1k (b)-adjuster".
class python_adjust
{
public:
  explicit python_adjust(contextt &_context);

  /// Walk every non-type symbol's IREP2 value. Returns true on error —
  /// specifically if the post-adjust strong invariant is violated (a
  /// member2t/index2t source or a constant_struct2t type still carries an
  /// unresolved `symbol_type2t` after resolution); false on success, mirroring
  /// `clang_c_adjust::adjust()`.
  bool adjust();

  /// Recursively visit `expr` and its sub-expressions, resolving transient
  /// `symbol_type2t` member2t/index2t sources to their followed aggregate type
  /// (the V.1k two-phase source invariant), and completing a by-name
  /// `constant_struct2t` literal (S2): follow + pad its type and insert
  /// zero-valued padding operands when missing. Note S2 resolves *eagerly*
  /// where the legacy adjust_struct leaves the literal's type lazily by-name
  /// (the deliberate RV-adj6 divergence — IREP2's strong construction
  /// invariant requires the resolved type on the node). Recurses operands
  /// first, so nested sources (`self.b.a`) resolve inner-to-outer.
  void adjust_expr(expr2tc &expr);

  /// IREP2-native `clang_c_adjust::adjust_type` (V.1k/B.5 milestone step S1):
  /// expand a macro `symbol_type2t` to the symbol's type, adjust an array's
  /// (VLA) size expression and element type, and complete a struct/union by
  /// recursing its member types and inserting alignment padding. Padding
  /// reuses the legacy `add_padding` through the type round-trip
  /// (`migrate_type_back` → `add_padding` → `migrate_type`) — lossless for
  /// every type the Python frontend emits (a packed *union* would drop its
  /// packed flag, but the converter emits no unions) — so the layout is
  /// byte-identical to the legacy pass by construction (risk RV-adj5). A
  /// non-macro tag reference deliberately stays by-name, exactly as the
  /// legacy pass leaves it (parity subtlety RV-adj6); IREP2 has no incomplete
  /// aggregates (an incomplete type stays a `symbol_type2t`), so the legacy
  /// `!type.incomplete()` guard has no analogue here.
  ///
  /// Known S1 scope limits vs the legacy pass, deliberate until later
  /// S-steps: (1) an unknown top-level type symbol is left by-name for the
  /// exit invariant instead of abort()ing; (2) no `vector_typet` arm (the
  /// Python frontend never emits vector types); (3) *type symbols themselves*
  /// are not adjusted — the legacy adjust() completes all `is_type` symbols
  /// first so resolution sees fixed-up tags; on the live pipeline
  /// `clang_cpp_adjust` still does that, and the B.5 flip must add the
  /// type-symbol pre-pass before this pass becomes the sole resolver.
  void adjust_type(type2tc &type);

protected:
  contextt &context;
  namespacet ns;

  /// If `source` is a member2t/index2t source carrying a transient
  /// `symbol_type2t`, follow it to the resolved struct/union/array and retype
  /// the node in place (returns true); otherwise leave it (returns false). The
  /// source is a plain `symbol2t` (the instance) or a `dereference2t` of a
  /// `pointer→tag-Cls` instance pointer — both arrive as a symbol_type2t source,
  /// since a member/index cannot be constructed over a raw pointer.
  bool resolve_source(expr2tc &source);

  /// Post-adjust strong-invariant probe (V.1k B.4): append to `out` one
  /// human-readable entry per unresolved node reachable from `expr` — a
  /// `member2t`/`index2t` source or `constant_struct2t` type still carrying a
  /// transient `symbol_type2t` (the three relaxed construction asserts,
  /// irep2_expr.h), or a resolved-struct literal whose operand count
  /// disagrees with its component list. adjust() logs these entries when the
  /// exit invariant fires — the per-node detail is the work-list the B.5-era
  /// resolution steps (S3+) drain. Recursive.
  void collect_unresolved_sources(
    const expr2tc &expr,
    std::vector<std::string> &out) const;
};
