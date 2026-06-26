#ifndef PYTHON_FRONTEND_PYTHON_ADJUST_H_
#define PYTHON_FRONTEND_PYTHON_ADJUST_H_

#include <irep2/irep2.h>
#include <util/context.h>
#include <util/namespace.h>

/**
 * IREP2-native adjuster for the Python frontend (Part V Phase V.1k design (b),
 * docs/irep2-migration.md). This is the resolve-then-build pass that replaces
 * the legacy `clang_cpp_adjust` round-trip on Python output: it walks the IREP2
 * body (`symbol.get_value2()`) and follows `symbol_type2t` member/index sources
 * to their resolved `struct_type2t`/`array_type2t` so that, post-adjust, every
 * `member2t`/`index2t` source satisfies the strong resolved-source invariant.
 *
 * Sub-phase status:
 *   B.0 — structural no-op walker (shipped).
 *   B.1 — member/index source following (this file): a `symbol_type2t` source is
 *         followed via `namespacet::follow` to its struct/array, recursively
 *         (resolve `X` before `X.a`). A pointer base is dereferenced by the
 *         converter before the node is built (the construction assert forbids a
 *         pointer source), so it never reaches the adjuster as a raw source.
 *         Still **not wired into** `python_languaget::final` (that is B.2), so
 *         it remains dead-but-tested — `unit/python-frontend/python_adjust_test.cpp`
 *         pins the traversal and the resolution.
 */
class python_adjust
{
public:
  explicit python_adjust(contextt &_context);
  virtual ~python_adjust() = default;

  /** Walk every non-type symbol's IREP2 body and write the resolved form back.
   *  Returns true on error (none today), mirroring `clang_c_adjust::adjust()`. */
  bool adjust();

protected:
  contextt &context;
  namespacet ns;

  /** Recursively resolve an IREP2 expression in place. Operands are resolved
   *  first (post-order), then a `member2t`/`index2t` source whose type is an
   *  unresolved `symbol_type2t` (directly or behind a pointer) is followed to
   *  its aggregate type. Virtual so tests can observe the traversal. */
  virtual void adjust_expr(expr2tc &expr);

private:
  /** If @p source carries an unresolved `symbol_type2t`, rewrite it to carry the
   *  followed aggregate type and return true. Otherwise leave it untouched and
   *  return false. */
  bool resolve_aggregate_source(expr2tc &source) const;
};

#endif /* PYTHON_FRONTEND_PYTHON_ADJUST_H_ */
