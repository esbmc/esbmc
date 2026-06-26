#ifndef PYTHON_FRONTEND_PYTHON_ADJUST_H_
#define PYTHON_FRONTEND_PYTHON_ADJUST_H_

#include <irep2/irep2.h>
#include <util/context.h>
#include <util/namespace.h>

/**
 * IREP2-native adjuster for the Python frontend (Part V Phase V.1k design (b),
 * docs/irep2-migration.md). This is the resolve-then-build pass that, in later
 * sub-phases, follows `symbol_type2t` member/index sources to their resolved
 * `struct_type2t`/`array_type2t` over the IREP2 body (`symbol.get_value2()`),
 * replacing the legacy `clang_cpp_adjust` round-trip on Python output.
 *
 * This file ships **B.0**: a structural no-op walker. It visits every operand
 * of every function body's IREP2 expression tree and changes nothing. It is not
 * wired into `python_languaget::final` yet, so it is dead-but-tested — the
 * traversal is exercised by `unit/python-frontend/python_adjust_test.cpp` so
 * that B.1 (member/index source following) has a verified walk to extend.
 */
class python_adjust
{
public:
  explicit python_adjust(contextt &_context);
  virtual ~python_adjust() = default;

  /** Walk every non-type symbol's IREP2 body. Returns true on error (none in
   *  B.0), mirroring `clang_c_adjust::adjust()`. */
  bool adjust();

protected:
  contextt &context;
  namespacet ns;

  /** Recursively visit an IREP2 expression. B.0: descends into every operand
   *  and performs no transformation. Virtual so later phases override the
   *  member/index source-following behaviour and so tests can observe the
   *  traversal. */
  virtual void adjust_expr(const expr2tc &expr);
};

#endif /* PYTHON_FRONTEND_PYTHON_ADJUST_H_ */
