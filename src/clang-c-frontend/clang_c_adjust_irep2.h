#pragma once

#include <util/symtab/context.h>
#include <irep2/irep2.h>

/// Phase 6 (C.3) IREP2-native adjuster for the C frontend.
///
/// The eventual replacement for `clang_c_adjust`, following the shape
/// `python_adjust` established (docs/roadmap/scope-clang-c-irep2.md §6): an
/// in-place recursive walk over `expr2tc` rather than the converter's
/// out-parameter seam.
///
/// At this stage the walk is deliberately **read-only**: it reads each code
/// symbol's IREP2 value and recurses, and never writes one back. That keeps the
/// pass inert by construction rather than by argument -- there is no write path
/// to be wrong -- while still exercising `migrate_expr` over every construct the
/// C corpus contains, since `symbolt::get_value2()` migrates the legacy value on
/// demand. A construct that cannot migrate aborts here instead of much later.
///
/// Read-only also side-steps the round-trip losses `python_adjust` documents
/// (a bitfield's `#bitfield` flag, an explicit alignment attribute): those only
/// matter to a write-back, and C headers are exactly the place they occur.
class clang_c_adjust_irep2
{
public:
  explicit clang_c_adjust_irep2(contextt &_context) : context(_context)
  {
  }

  /// Walk every code symbol's IREP2 value. Returns false; there is no failure
  /// mode yet, and the signature matches `clang_c_adjust::adjust()` so the
  /// driver can call either.
  bool adjust();

  void adjust_expr(expr2tc &expr);

private:
  contextt &context;
};
