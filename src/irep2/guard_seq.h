#ifndef IREP2_GUARD_SEQ_H_
#define IREP2_GUARD_SEQ_H_

#include <cstddef>
#include <immer/flex_vector.hpp>
#include <immer/memory_policy.hpp>
#include <irep2/irep2_expr.h>

/** Oldest-first immutable sequence of guard conjuncts with O(1) copy.
 *
 *  Replaces the flat `std::vector<expr2tc>` that `guard2tc` used to hold.
 *  The vector was deep-copied on every guard copy (O(N) element copies plus
 *  N refcount bumps on the conjuncts) and rebuilt from scratch on every set
 *  operation, giving the Θ(N²) blow-up at deep loop unwinding.
 *
 *  Backed by `immer::flex_vector`, a persistent (structurally-shared) RRB
 *  vector. Conjuncts stay oldest-first — the same canonical order as
 *  guard2tc's cached left-leaning and-chain, so the index correspondence the
 *  prefix/diff logic relies on is preserved exactly.
 *
 *  Costs: copy O(1) (shared root); push_back / index O(log32 N) ≈ O(1) for
 *  the few-hundred-element guards symex builds; prefix/suffix (take/drop)
 *  share whole subtrees; iterate O(N).
 *
 *  Ownership: immer's default (atomic-refcounted) memory policy. Guards are
 *  in practice thread-confined — the only symex parallelism
 *  (`--parallel-solving`) forks threads to *solve* an already-built equation,
 *  while the guard set-algebra runs in the single-threaded symex phase, and
 *  the conjunct list is never stored in the equation (only the expr2tc base
 *  is). A non-atomic refcount would therefore be sound today, but it would be
 *  load-bearing on "symex never parallelizes"; the ATOMIC refcount removes
 *  that latent-race footgun.
 *
 *  We keep the atomic refcount but drop the spinlock: immer's *default*
 *  memory policy pairs the atomic refcount with `spinlock_policy`, which
 *  takes a test_and_set lock around tree operations to make immer's
 *  *transient* (batch-mutation) API safe to share across threads. We never
 *  share a transient across threads, and the lock measured as a 2× symex
 *  regression (it fires on every node visit in the hot rrbtree descent). So
 *  the policy is atomic `refcount_policy` + `no_lock_policy`: thread-safe
 *  refcounting, no per-operation lock. */
class guard_seq
{
  using memory_policy = immer::memory_policy<
    immer::default_heap_policy,
    immer::refcount_policy,
    immer::no_lock_policy>;
  using vector_t = immer::flex_vector<expr2tc, memory_policy>;

  vector_t v_;

  explicit guard_seq(vector_t v) : v_(std::move(v))
  {
  }

public:
  guard_seq() = default;

  std::size_t size() const
  {
    return v_.size();
  }
  bool empty() const
  {
    return v_.empty();
  }

  // Value-returning, not reference-returning: a reference into immer storage
  // is only valid until the next push_back/clear replaces the root. Returning
  // by value (an expr2tc copy = one refcount bump) makes it impossible for a
  // caller to hold a dangling element ref across a guard mutation.
  expr2tc operator[](std::size_t i) const
  {
    return v_[i];
  }
  expr2tc front() const
  {
    return v_.front();
  }
  expr2tc back() const
  {
    return v_.back();
  }

  void push_back(const expr2tc &e)
  {
    v_ = std::move(v_).push_back(e);
  }

  /** Logical prefix [0, n) — shares subtrees, no element copies. */
  guard_seq prefix(std::size_t n) const
  {
    return guard_seq(v_.take(n));
  }

  /** Logical suffix [n, size) — shares subtrees, no element copies. */
  guard_seq suffix(std::size_t n) const
  {
    return guard_seq(v_.drop(n));
  }

  void clear()
  {
    v_ = vector_t{};
  }

  using const_iterator = vector_t::const_iterator;
  const_iterator begin() const
  {
    return v_.begin();
  }
  const_iterator end() const
  {
    return v_.end();
  }

  bool operator==(const guard_seq &o) const
  {
    return v_ == o.v_;
  }
  bool operator!=(const guard_seq &o) const
  {
    return v_ != o.v_;
  }
};

#endif /* IREP2_GUARD_SEQ_H_ */
