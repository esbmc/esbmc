#ifndef IREP2_GUARD_SEQ_H_
#define IREP2_GUARD_SEQ_H_

#include <cstddef>
#include <immer/map.hpp>
#include <immer/vector.hpp>
#include <immer/memory_policy.hpp>
#include <irep2/irep2_expr.h>

/** Oldest-first immutable sequence of guard conjuncts with O(1) copy.
 *
 *  Replaces the flat `std::vector<expr2tc>` that `guard2tc` used to hold.
 *  The vector was deep-copied on every guard copy (O(N) element copies plus
 *  N refcount bumps on the conjuncts) and rebuilt from scratch on every set
 *  operation, giving the Θ(N²) blow-up at deep loop unwinding.
 *
 *  Backed by `immer::vector`, a persistent (structurally-shared) bit-mapped
 *  vector trie. Conjuncts stay oldest-first — the same canonical order as
 *  guard2tc's cached left-leaning and-chain, so the index correspondence the
 *  prefix/diff logic relies on is preserved exactly.
 *
 *  Costs: copy O(1) (shared root); push_back / index O(log32 N) ≈ O(1) for
 *  the few-hundred-element guards symex builds; prefix (take) shares subtrees;
 *  iterate O(N). We use immer::vector rather than flex_vector (the relaxed
 *  RRB tree): flex's only edge is O(log N) drop()/suffix slicing, but measured
 *  max suffix length in real symex is 1 (guards diverge one conjunct at a time
 *  at merge points), so flex's relaxed-radix bookkeeping is pure overhead.
 *  A/B: immer::vector was ~5% fewer instructions and ~15% faster symex on
 *  test_locks_13, same verdicts. suffix() rebuilds [n,size) since plain vector
 *  has no drop(); that is O(suffix) but suffix is ~1 in practice.
 *
 *  Ownership: NON-atomic refcount, no lock (`unsafe_refcount_policy` +
 *  `no_lock_policy`). Guards are thread-confined — the only symex parallelism
 *  (`--parallel-solving`) forks threads to *solve* an already-built equation,
 *  while the guard set-algebra runs in the single-threaded symex phase, and
 *  the conjunct list is never stored in the equation (only the expr2tc base
 *  is, and that is an atomic irep_container). So a non-atomic refcount is
 *  sound. NOTE: this is load-bearing on "symex is not parallelized" — if that
 *  ever changes, switch to `refcount_policy` (atomic; measured ~10-15% slower
 *  here, not the 2× the value-returning accessors below cost). immer's
 *  *default* policy additionally pairs the atomic refcount with
 *  `spinlock_policy` (a test_and_set lock around every tree op for the
 *  transient API); we never share transients across threads, so `no_lock` is
 *  used regardless of refcount choice. */
class guard_seq
{
  using memory_policy = immer::memory_policy<
    immer::default_heap_policy,
    immer::unsafe_refcount_policy,
    immer::no_lock_policy>;
  using vector_t = immer::vector<expr2tc, memory_policy>;
  // Structural hash of a conjunct -> occurrences in v_. A count, not a flag,
  // so that prefix() can drop a hash that also occurs inside the kept prefix.
  using index_t = immer::map<
    std::size_t,
    std::size_t,
    std::hash<std::size_t>,
    std::equal_to<std::size_t>,
    memory_policy>;

  vector_t v_;
  index_t index_;

  guard_seq(vector_t v, index_t index)
    : v_(std::move(v)), index_(std::move(index))
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

  // Reference-returning into immer storage. The reference is valid only
  // until the next push_back()/clear() replaces the root — callers must not
  // stash an element reference across a guard mutation. (Value-returning was
  // measured at ~2x symex: an expr2tc copy is an atomic refcount bump, and
  // the -=/|=/build_guard_expr/== loops access elements O(N) times per merge,
  // so the per-access copy dominates.) All internal call sites read the
  // reference before any mutation; this is the documented precondition.
  const expr2tc &operator[](std::size_t i) const
  {
    return v_[i];
  }
  const expr2tc &front() const
  {
    return v_.front();
  }
  const expr2tc &back() const
  {
    return v_.back();
  }

  void push_back(const expr2tc &e)
  {
    v_ = std::move(v_).push_back(e);
    const std::size_t h = irep2_hash()(e);
    const std::size_t *n = index_.find(h);
    index_ = std::move(index_).set(h, n ? *n + 1 : 1);
  }

  /** False only if no conjunct is structurally equal to `e`; true may be a
   *  hash collision. O(log32 N). */
  bool may_contain(const expr2tc &e) const
  {
    return index_.find(irep2_hash()(e)) != nullptr;
  }

  /** Logical prefix [0, n) — shares subtrees, no element copies. */
  guard_seq prefix(std::size_t n) const
  {
    if (n >= v_.size())
      return *this;
    index_t index = index_;
    for (std::size_t i = n; i < v_.size(); ++i)
    {
      const std::size_t h = irep2_hash()(v_[i]);
      const std::size_t c = *index.find(h);
      index =
        c == 1 ? std::move(index).erase(h) : std::move(index).set(h, c - 1);
    }
    return guard_seq(v_.take(n), std::move(index));
  }

  /** Logical suffix [n, size). immer::vector has no drop(), so rebuild
   *  [n,size) element-wise. O(suffix), but the suffix length in symex guard
   *  subtraction is ~1 in practice (guards diverge one conjunct at a time at
   *  merge points), so this is cheaper than flex_vector's relaxed-radix drop.
   */
  guard_seq suffix(std::size_t n) const
  {
    guard_seq r;
    for (std::size_t i = n; i < v_.size(); ++i)
      r.push_back(v_[i]);
    return r;
  }

  void clear()
  {
    v_ = vector_t{};
    index_ = index_t{};
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
