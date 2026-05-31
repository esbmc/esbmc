#ifndef IREP2_GUARD_H_
#define IREP2_GUARD_H_

#include <vector>
#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>

/** Path-condition guard container.
 *
 *  `guard2tc` extends `expr2tc` with a flat list of conjuncts
 *  (`guard_list`) accumulated along an execution path. Wherever an
 *  `expr2tc` is expected, a `guard2tc` works directly because it *is*
 *  an `expr2tc`.
 *
 *  Mutators maintain the expr2tc base incrementally as the and-chain
 *  of the current conjuncts (matching the historic
 *  guardt::g_expr/build_guard_expr split), so as_expr() and implicit
 *  conversions are pure reads. A default-constructed guard has an
 *  empty list and a nil base; as_expr() then short-circuits to
 *  `gen_true_expr()` without touching the base.
 *
 *  Invariant: `guard_list` is treated as a set of conjuncts — no
 *  duplicates are expected since `x && x ≡ x`. operator-= and
 *  operator|= use hash-set membership probes which give set rather
 *  than multiset semantics; symex's natural pattern of adding fresh
 *  branch symbols preserves uniqueness. Injecting the same conjunct
 *  twice leaves a logically-redundant duplicate in the list and
 *  weakens the set-op equality guarantees but does not produce
 *  unsound path conditions. */
class guard2tc : public expr2tc
{
public:
  std::vector<expr2tc> guard_list;

  guard2tc() = default;
  guard2tc(const guard2tc &) = default;
  guard2tc(guard2tc &&) noexcept = default;
  guard2tc &operator=(const guard2tc &) = default;
  guard2tc &operator=(guard2tc &&) noexcept = default;

  void add(const expr2tc &expr);
  void append(const guard2tc &other);

  expr2tc as_expr() const;
  void guard_expr(expr2tc &dest) const;

  bool disjunction_may_simplify(const guard2tc &other) const;
  bool is_true() const;
  bool is_false() const;

  void make_true();
  void make_false();

  /** Deleted. The inherited `irep_container::swap` would silently
   *  swap only the expr2tc base and leave `guard_list` desynced —
   *  use `g = std::move(other)` instead, which moves both members
   *  atomically and is one-way (cheaper than swap when the source
   *  goes out of scope immediately after). */
  void swap(guard2tc &) = delete;

  void dump() const;

private:
  bool is_single_symbol() const;
  void clear();

  /** Append a single non-and2t conjunct, applying the same trivial
   *  absorptions add() does. Used by the fast path in add() and by
   *  the slow-path worklist after unfolding nested and2ts to leaves. */
  void add_leaf(const expr2tc &expr);

  /** Build the and-chain into the expr2tc base from guard_list. Used
   *  after operator-= / operator|= installs a fresh list. Asserts that
   *  the base is nil (callers reset it first). */
  void build_guard_expr();
  void set_guard_list_and_rebuild(std::vector<expr2tc> &&new_guard_list);
  void set_guard_list_and_base(
    std::vector<expr2tc> &&new_guard_list,
    const expr2tc &base);

  friend guard2tc &operator-=(guard2tc &g1, const guard2tc &g2);
  friend guard2tc &operator|=(guard2tc &g1, const guard2tc &g2);
  friend bool operator==(const guard2tc &g1, const guard2tc &g2);
};

#endif /* IREP2_GUARD_H_ */
