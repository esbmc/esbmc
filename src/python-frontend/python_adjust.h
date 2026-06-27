#pragma once

#include <util/context.h>
#include <util/namespace.h>
#include <irep2/irep2.h>

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

  /// Walk every non-type symbol's IREP2 value. Returns false on success,
  /// mirroring `clang_c_adjust::adjust()`.
  bool adjust();

  /// Recursively visit `expr` and its sub-expressions, resolving transient
  /// `symbol_type2t` member2t/index2t sources to their followed aggregate type
  /// (the V.1k two-phase source invariant). Recurses operands first, so nested
  /// sources (`self.b.a`) resolve inner-to-outer.
  void adjust_expr(expr2tc &expr);

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
};
