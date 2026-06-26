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

  /// Recursively visit `expr` and its sub-expressions. B.0: structural no-op.
  /// Phase B.1 hooks member2t/index2t source resolution in here.
  void adjust_expr(expr2tc &expr);

protected:
  contextt &context;
  namespacet ns;
};
