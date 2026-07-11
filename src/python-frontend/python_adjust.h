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

  /// Two-phase walk mirroring `clang_c_adjust::adjust()`: first complete
  /// every type symbol's IREP2 type via adjust_type (macro expansion,
  /// padding), so the value resolution below always follows fixed-up tags;
  /// then walk every code symbol's IREP2 value. Returns true on error —
  /// specifically if the post-adjust strong invariant is violated (a
  /// member2t/index2t source or a constant_struct2t type still carries an
  /// unresolved `symbol_type2t` after resolution, or a resolved literal's
  /// operand count disagrees with its component list); false on success.
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
  /// Known scope limits vs the legacy pass, deliberate until later S-steps:
  /// (1) an unknown top-level type symbol is left by-name for the exit
  /// invariant instead of abort()ing; (2) no `vector_typet` arm (the Python
  /// frontend never emits vector types); (3) non-type symbols' *own* types
  /// (e.g. a function's code type) are not adjusted — the legacy
  /// adjust_symbol completes them (`clang_c_adjust_expr.cpp:70-74`), and
  /// `clang_cpp_adjust` still does on the live pipeline. Type symbols ARE
  /// completed: adjust() runs a type-symbol pre-pass before value
  /// resolution, mirroring the legacy two-phase order. (4) The pre-pass
  /// write-back (`set_type(type2tc)`) cannot carry legacy-only struct
  /// metadata — the `"bases"` sub-irep (read by exception_typeid.cpp and
  /// base_type.cpp for Python exception-hierarchy/catch matching) and
  /// component `access`/`#is_padding` flags are lost if the write-back
  /// fires. Inert today (the write-back never fires post-clang_cpp_adjust);
  /// the B.5 flip must either re-attach the preserved sub-ireps on a legacy
  /// write-back or move the `"bases"` carriage to IREP2 (W3/V.2) first.
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

  /// If `fn` is a symbol whose symbol-table type is pointer-to-code — the
  /// lambda/def-alias call variable — re-type it from the table, wrap it in
  /// a dereference onto the followed code type (the legacy adjust_symbol +
  /// implicit-deref pair, clang_c_adjust_expr.cpp:918-926), and cast each
  /// argument to its declared parameter type (the legacy
  /// adjust_function_call_arguments analogue). Returns true when rewritten.
  bool wrap_function_pointer_callee(expr2tc &fn, std::vector<expr2tc> &args);

  /// Derive a cpp-throw's exception-id chain from its operand's type,
  /// mirroring `clang_cpp_adjust::convert_exception_id`: the bare class name
  /// followed by its direct bases for a class operand (both the by-name
  /// `symbol_type2t` tag and the S2-resolved `struct_type2t` shape),
  /// "void_ptr" for the untypeable-raise `any_type()` operand, a `_ptr`
  /// suffix through real pointers, and — like legacy — a synthetic
  /// type-id fallback so the result is never empty (remove_exceptions
  /// dereferences front()). Used by adjust_expr to complete an empty
  /// `code_cpp_throw2t::exception_list` — flip blocker #1
  /// (docs/irep2-migration.md, "Flip-probe census").
  std::vector<irep_idt> derive_exception_ids(const type2tc &type) const;

  /// Recursive worker for derive_exception_ids, threading the legacy `_ptr`
  /// suffix accumulation.
  void derive_exception_ids_rec(
    const type2tc &type,
    const std::string &suffix,
    std::vector<irep_idt> &ids) const;

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
