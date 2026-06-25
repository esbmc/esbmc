#pragma once

#include <util/context.h>
#include <util/namespace.h>
#include <irep2/irep2.h>

/// V.1k (b) IREP2-native Python adjuster — Phase B.0 skeleton.
///
/// This pass exists to replace, for the Python frontend, the legacy
/// `clang_cpp_adjust` round-trip currently run on Python output
/// (`python_language.cpp`). It walks each non-type symbol's IREP2 value
/// (`symbolt::get_value2()`) recursively.
///
/// **B.0 is a structural no-op:** it visits every node and leaves the tree
/// byte-identical. It is deliberately *not* wired into `python_language.cpp`
/// yet — it is dead-but-tested, mirroring the "add the machinery, prove it
/// inert, wire it later" pattern used for the V.4.0 structured-CF kinds
/// (esbmc/esbmc#5265) and the V-track back-migration arms (#4737).
///
/// Later phases give `adjust_expr` real behaviour: following a transient
/// `symbol_type2t` `member2t`/`index2t` source to its resolved
/// `struct_type2t`/`array_type2t` (the V.1k "two-phase source invariant"),
/// pointer auto-deref, and `#cpp_type`/`#member_name` carriage — at which
/// point the legacy `clang_cpp_adjust` hop on Python output is dropped.
/// See `docs/irep2-migration.md`, section "V.1k (b)-adjuster".
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
