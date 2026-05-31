#include <algorithm>
#include <unordered_set>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_utils.h>

namespace
{
bool same_cached_expr(const guard2tc &g1, const guard2tc &g2)
{
  return static_cast<const expr2tc &>(g1).get() ==
         static_cast<const expr2tc &>(g2).get();
}

const expr2t *expr_ptr(const expr2tc &expr)
{
  return expr.get();
}

std::size_t common_pointer_prefix_size(const guard2tc &g1, const guard2tc &g2)
{
  std::size_t prefix_size = 0;
  const std::size_t min_size =
    std::min(g1.guard_list.size(), g2.guard_list.size());
  while (prefix_size < min_size && expr_ptr(g1.guard_list[prefix_size]) ==
                                     expr_ptr(g2.guard_list[prefix_size]))
    ++prefix_size;
  return prefix_size;
}

#ifndef NDEBUG
// cached_prefix_expr decides the prefix relationship from the cached
// and-chain base alone (a pointer walk down side_1). That is sound only
// because build_guard_expr produces a canonical left-leaning chain, so
// base equality implies the flattened conjunct sequences match
// element-wise. This predicate makes that implicit invariant explicit:
// callers that slice guard_list at prefix.size() on the strength of a
// cached_prefix_expr hit assert it, turning a hypothetical hash-cons
// aliasing corruption into a loud test failure instead of a silently
// wrong path condition. Debug-only; compiled out of release builds.
bool guard_list_prefix_matches(const guard2tc &prefix, const guard2tc &guard)
{
  if (prefix.guard_list.size() > guard.guard_list.size())
    return false;
  for (std::size_t i = 0; i < prefix.guard_list.size(); ++i)
    if (expr_ptr(prefix.guard_list[i]) != expr_ptr(guard.guard_list[i]))
      return false;
  return true;
}
#endif

bool cached_prefix_expr(
  const guard2tc &prefix,
  const guard2tc &guard,
  expr2tc &prefix_expr)
{
  const std::size_t prefix_size = prefix.guard_list.size();
  const std::size_t guard_size = guard.guard_list.size();
  if (prefix_size > guard_size)
    return false;

  if (prefix_size == 0)
    return true;

  if (prefix_size == guard_size)
  {
    if (!same_cached_expr(prefix, guard))
      return false;
    prefix_expr = prefix;
    assert(!is_nil_expr(prefix_expr));
    return true;
  }

  expr2tc cur = guard;
  for (std::size_t i = guard_size - prefix_size; i != 0; --i)
  {
    if (!is_and2t(cur))
      return false;
    cur = to_and2t(cur).side_1;
  }

  if (
    static_cast<const expr2tc &>(cur).get() !=
    static_cast<const expr2tc &>(prefix).get())
    return false;

  prefix_expr = cur;
  assert(!is_nil_expr(prefix_expr));
  return true;
}

bool cached_prefix_expr_at(
  const guard2tc &guard,
  const std::size_t prefix_size,
  expr2tc &prefix_expr)
{
  const std::size_t guard_size = guard.guard_list.size();
  if (prefix_size > guard_size)
    return false;

  if (prefix_size == 0)
  {
    prefix_expr = expr2tc();
    return true;
  }

  if (prefix_size == guard_size)
  {
    prefix_expr = guard;
    return true;
  }

  expr2tc cur = guard;
  for (std::size_t i = guard_size - prefix_size; i != 0; --i)
  {
    if (!is_and2t(cur))
      return false;
    cur = to_and2t(cur).side_1;
  }

  prefix_expr = cur;
  return true;
}
} // namespace

expr2tc guard2tc::as_expr() const
{
  // Mutators maintain the expr2tc base so it already matches the
  // current chain for size >= 1: single conjunct → that conjunct,
  // multi → and-chain. Only the empty-list case needs a translation
  // (base stays nil so default-constructed guards don't allocate).
  if (is_true())
    return gen_true_expr();
  return *this;
}

void guard2tc::add(const expr2tc &expr)
{
  // Fast path: the input is a non-and2t leaf. No worklist needed —
  // skip straight to the single-conjunct insertion. This is the
  // overwhelmingly common case (typical add() call in symex passes
  // a freshly-computed branch condition that's a symbol, equality,
  // or other non-and leaf).
  if (!is_and2t(expr))
  {
    add_leaf(expr);
    return;
  }

  // Slow path: input has nested and2ts to unfold. Use a heap
  // worklist instead of recursion so a deep left-leaning chain
  // (which symex can build) doesn't blow the thread stack at ~10k+
  // levels. Depth-first, left-side first, so leaf order matches
  // the historic recursive version.
  std::vector<expr2tc> worklist;
  worklist.push_back(expr);
  while (!worklist.empty())
  {
    expr2tc cur = std::move(worklist.back());
    worklist.pop_back();

    if (is_and2t(cur))
    {
      const and2t &theand = to_and2t(cur);
      worklist.push_back(theand.side_2);
      worklist.push_back(theand.side_1);
      continue;
    }
    add_leaf(cur);
  }
}

void guard2tc::add_leaf(const expr2tc &expr)
{
  // Invariant on entry: `expr` is not an and2t. Apply the trivial
  // absorptions and append.
  if (is_false() || ::is_true(expr))
    return;

  if (is_true() || ::is_false(expr))
    clear();

  guard_list.push_back(expr);

  // Update the cached and-chain incrementally.
  if (is_nil_expr(*this))
  {
    assert(guard_list.size() == 1);
    expr2tc::operator=(expr);
  }
  else
  {
    expr2tc::operator=(and2tc(*this, expr));
  }
}

void guard2tc::guard_expr(expr2tc &dest) const
{
  if (is_true())
    return;

  if (::is_false(dest))
  {
    dest = as_expr();
    make_not(dest);
    return;
  }

  dest = implies2tc(as_expr(), dest);
}

void guard2tc::build_guard_expr()
{
  // Only used after operator-= / operator|= installs a fresh list and
  // resets the base; matches the original guardt::build_guard_expr.
  assert(is_nil_expr(*this));

  if (is_true())
    return;

  if (is_single_symbol())
  {
    expr2tc::operator=(guard_list.front());
    return;
  }

  auto it = guard_list.begin();
  expr2tc arg1 = *it++;
  expr2tc arg2 = *it++;
  expr2tc res = and2tc(arg1, arg2);
  while (it != guard_list.end())
    res = and2tc(res, *it++);
  expr2tc::operator=(res);
}

void guard2tc::set_guard_list_and_rebuild(std::vector<expr2tc> &&new_guard_list)
{
  clear();
  guard_list = std::move(new_guard_list);
  build_guard_expr();
}

void guard2tc::append(const guard2tc &other)
{
  // Reserve up-front so the per-add() push_back doesn't trigger a
  // geometric reallocation in the middle of the loop. The bound is
  // exact when `other` has no nested and2ts and no true/false
  // sentinels — those cases would push less. Caller must not pass
  // *this as `other`; vector growth would invalidate the range-for.
  guard_list.reserve(guard_list.size() + other.guard_list.size());
  for (const auto &c : other.guard_list)
    add(c);
}

guard2tc &operator-=(guard2tc &g1, const guard2tc &g2)
{
  if (g1.is_true() || g2.is_true())
    return g1;

  std::size_t prefix_size = 0;
  const std::size_t min_size =
    std::min(g1.guard_list.size(), g2.guard_list.size());
  while (prefix_size < min_size && expr_ptr(g1.guard_list[prefix_size]) ==
                                     expr_ptr(g2.guard_list[prefix_size]))
    ++prefix_size;

  if (prefix_size != 0)
  {
    if (prefix_size == g1.guard_list.size())
    {
      g1.make_true();
      return g1;
    }

    if (prefix_size == g2.guard_list.size())
    {
      std::vector<expr2tc> diff(
        g1.guard_list.begin() + prefix_size, g1.guard_list.end());
      g1.set_guard_list_and_rebuild(std::move(diff));
      return g1;
    }

    std::unordered_set<const expr2t *> g2_suffix;
    g2_suffix.reserve(g2.guard_list.size() - prefix_size);
    for (auto it = g2.guard_list.begin() + prefix_size;
         it != g2.guard_list.end();
         ++it)
      g2_suffix.insert(expr_ptr(*it));

    std::unordered_set<expr2tc, irep2_hash> g2_suffix_exprs;
    bool built_g2_suffix_exprs = false;

    std::vector<expr2tc> diff;
    diff.reserve(g1.guard_list.size() - prefix_size);
    for (auto it = g1.guard_list.begin() + prefix_size;
         it != g1.guard_list.end();
         ++it)
    {
      if (g2_suffix.find(expr_ptr(*it)) != g2_suffix.end())
        continue;

      if (!built_g2_suffix_exprs)
      {
        g2_suffix_exprs.reserve(g2.guard_list.size() - prefix_size);
        g2_suffix_exprs.insert(
          g2.guard_list.begin() + prefix_size, g2.guard_list.end());
        built_g2_suffix_exprs = true;
      }

      if (g2_suffix_exprs.find(*it) == g2_suffix_exprs.end())
        diff.push_back(*it);
    }

    g1.set_guard_list_and_rebuild(std::move(diff));
    return g1;
  }

  // Hot symex phi path: tmp_guard is commonly a guard that was copied
  // from cur_state->guard and then extended by one branch conjunct. The
  // cached and-chain preserves that copy/append relation structurally:
  //   extended.base == and2tc(prefix.base, new_conjunct)
  // Detect it before the hash-set fallback so g1 -= prefix builds only
  // the small suffix instead of scanning and rebuilding the whole guard.
  expr2tc prefix_expr;
  if (cached_prefix_expr(g2, g1, prefix_expr))
  {
    // g2 is a cached prefix of g1: its conjuncts occupy g1.guard_list's
    // first |g2| positions, so the difference is g1's suffix.
    assert(guard_list_prefix_matches(g2, g1));

    if (g1.guard_list.size() == g2.guard_list.size())
    {
      g1.make_true();
      return g1;
    }

    std::vector<expr2tc> diff(
      g1.guard_list.begin() + g2.guard_list.size(), g1.guard_list.end());
    g1.set_guard_list_and_rebuild(std::move(diff));
    return g1;
  }

  if (cached_prefix_expr(g1, g2, prefix_expr))
  {
    g1.make_true();
    return g1;
  }

  // Set difference by hashed membership: result is conjuncts in g1
  // not in g2. Order-independent, so it stays correct regardless of
  // how the two guards' lists were built up (which std::set_difference
  // is NOT — it requires sorted ranges, and guard_list is in
  // insertion order).
  std::unordered_set<expr2tc, irep2_hash> g2_set(
    g2.guard_list.begin(), g2.guard_list.end());

  std::vector<expr2tc> diff;
  diff.reserve(g1.guard_list.size());
  for (const auto &c : g1.guard_list)
    if (g2_set.find(c) == g2_set.end())
      diff.push_back(c);

  g1.set_guard_list_and_rebuild(std::move(diff));
  return g1;
}

guard2tc &operator|=(guard2tc &g1, const guard2tc &g2)
{
  // Trivial absorptions.
  if (g2.is_false() || g1.is_true())
    return g1;
  if (g1.is_false() || g2.is_true())
  {
    g1 = g2;
    return g1;
  }

  // Fast structural subsumption before the set-algebra fallback. Guards
  // grown by copy-then-add share the old cached and-chain as the left
  // operand of the new chain, so ordered prefix cases can be detected by
  // walking only the appended suffix.
  expr2tc prefix_expr;
  if (cached_prefix_expr(g1, g2, prefix_expr))
    return g1;

  if (cached_prefix_expr(g2, g1, prefix_expr))
  {
    // g2 ⊆ g1 (g2 is a cached prefix), so g1 ⇒ g2 and g1 || g2 ≡ g2.
    assert(guard_list_prefix_matches(g2, g1));
    std::vector<expr2tc> prefix_list(g2.guard_list);
    g1.set_guard_list_and_rebuild(std::move(prefix_list));
    return g1;
  }

  std::size_t prefix_size = common_pointer_prefix_size(g1, g2);

  if (prefix_size != 0)
  {
    expr2tc prefix_expr;
    if (cached_prefix_expr_at(g1, prefix_size, prefix_expr))
    {
      if (prefix_size == g1.guard_list.size())
        return g1;

      if (prefix_size == g2.guard_list.size())
      {
        std::vector<expr2tc> prefix_list(
          g1.guard_list.begin(), g1.guard_list.begin() + prefix_size);
        g1.set_guard_list_and_rebuild(std::move(prefix_list));
        return g1;
      }

      std::unordered_set<const expr2t *> g2_suffix;
      g2_suffix.reserve(g2.guard_list.size() - prefix_size);
      for (auto it = g2.guard_list.begin() + prefix_size;
           it != g2.guard_list.end();
           ++it)
        g2_suffix.insert(expr_ptr(*it));

      std::vector<expr2tc> common_suffix;
      std::vector<expr2tc> new_g1_list;
      common_suffix.reserve(g1.guard_list.size() - prefix_size);
      new_g1_list.reserve(g1.guard_list.size() - prefix_size);
      for (auto it = g1.guard_list.begin() + prefix_size;
           it != g1.guard_list.end();
           ++it)
      {
        if (g2_suffix.erase(expr_ptr(*it)))
          common_suffix.push_back(*it);
        else
          new_g1_list.push_back(*it);
      }

      if (new_g1_list.empty())
        return g1;

      std::vector<expr2tc> new_g2_list;
      new_g2_list.reserve(g2_suffix.size());
      for (auto it = g2.guard_list.begin() + prefix_size;
           it != g2.guard_list.end();
           ++it)
        if (g2_suffix.count(expr_ptr(*it)))
          new_g2_list.push_back(*it);

      std::vector<expr2tc> merged_list(
        g1.guard_list.begin(), g1.guard_list.begin() + prefix_size);
      g1.set_guard_list_and_rebuild(std::move(merged_list));
      for (const auto &c : common_suffix)
        g1.add(c);

      if (new_g2_list.empty())
        return g1;

      guard2tc new_g1;
      new_g1.guard_list.swap(new_g1_list);
      new_g1.build_guard_expr();

      guard2tc new_g2;
      new_g2.guard_list.swap(new_g2_list);
      new_g2.build_guard_expr();

      expr2tc or_expr = or2tc(new_g1.as_expr(), new_g2.as_expr());
      if (new_g1.is_single_symbol() && new_g2.is_single_symbol())
        simplify(or_expr);

      g1.add(or_expr);
      return g1;
    }
  }

  // g1 || g2 == common && (g1' || g2'), where common is the conjunct
  // intersection and g1'/g2' are the residuals. Guard conjuncts are
  // hash-consed, so pointer identity is the cheap membership key here:
  // it avoids re-hashing long expression trees on every loop-back merge.
  guard2tc common;
  guard2tc new_g1;
  guard2tc new_g2;
  common.guard_list.reserve(
    std::min(g1.guard_list.size(), g2.guard_list.size()));
  new_g1.guard_list.reserve(g1.guard_list.size());
  new_g2.guard_list.reserve(g2.guard_list.size());

  if (g2.guard_list.size() <= g1.guard_list.size())
  {
    std::unordered_set<const expr2t *> g2_remaining;
    g2_remaining.reserve(g2.guard_list.size());
    for (const auto &c : g2.guard_list)
      g2_remaining.insert(expr_ptr(c));

    for (const auto &c : g1.guard_list)
    {
      if (g2_remaining.erase(expr_ptr(c)))
        common.guard_list.push_back(c);
      else
        new_g1.guard_list.push_back(c);
    }

    for (const auto &c : g2.guard_list)
      if (g2_remaining.count(expr_ptr(c)))
        new_g2.guard_list.push_back(c);
  }
  else
  {
    std::unordered_set<const expr2t *> g1_remaining;
    g1_remaining.reserve(g1.guard_list.size());
    for (const auto &c : g1.guard_list)
      g1_remaining.insert(expr_ptr(c));

    for (const auto &c : g2.guard_list)
      if (!g1_remaining.erase(expr_ptr(c)))
        new_g2.guard_list.push_back(c);

    for (const auto &c : g1.guard_list)
    {
      if (g1_remaining.count(expr_ptr(c)))
        new_g1.guard_list.push_back(c);
      else
        common.guard_list.push_back(c);
    }
  }

  if (common.is_true())
  {
    // Pointer identity is the fast path for symex guards, but preserve the
    // historical set semantics for independently-created equal expressions.
    std::unordered_set<expr2tc, irep2_hash> g2_set(
      g2.guard_list.begin(), g2.guard_list.end());

    new_g1.guard_list.clear();
    new_g2.guard_list.clear();
    for (const auto &c : g1.guard_list)
    {
      if (g2_set.erase(c))
        common.guard_list.push_back(c);
      else
        new_g1.guard_list.push_back(c);
    }

    for (const auto &c : g2.guard_list)
      if (g2_set.count(c))
        new_g2.guard_list.push_back(c);
  }

  // If either residual is empty, that side equals `common` itself, so
  // the disjunction reduces to `common || (common && other) ≡ common`.
  // Skip the or2tc construction and chain-extend entirely. Covers the
  // pathological case where one guard is a prefix of the other (a
  // common pattern at branch joins where one side adds extra
  // conjuncts), and the identical-set case. This is the hot path at
  // loop-back merges: one guard is the other plus a fresh branch
  // conjunct, so exactly one residual is empty.
  //
  // The whole point of this branch is to avoid an O(N) rebuild. When
  // `common` has the same conjuncts as `g1` (i.e. nothing went into
  // new_g1), g1's cached and-chain is already correct — keep it instead
  // of rebuilding common's chain from scratch. That turns the dominant
  // merge case from O(N) to O(1) on the base, which is what makes deep
  // unwinding scale linearly rather than quadratically.
  if (new_g1.is_true())
    return g1; // common == g1, base unchanged

  if (new_g2.is_true())
  {
    // common == g2; g1 must drop the conjuncts that were unique to it.
    // common is a strict subset of g1, so its chain is shorter — build
    // it once and install.
    common.build_guard_expr();
    g1 = std::move(common);
    return g1;
  }

  common.build_guard_expr();
  new_g1.build_guard_expr();
  new_g2.build_guard_expr();

  // When both residuals are atomic, the OR may simplify trivially
  // (e.g. `a || !a` → true); ask the simplifier.
  expr2tc or_expr = or2tc(new_g1.as_expr(), new_g2.as_expr());
  if (new_g1.is_single_symbol() && new_g2.is_single_symbol())
    simplify(or_expr);

  // common already has its and-chain materialised; move it into g1
  // (transfers vector + cached chain without per-element refcount
  // churn) then extend with the OR of the residuals.
  g1 = std::move(common);
  g1.add(or_expr);
  return g1;
}

void guard2tc::dump() const
{
  for (const auto &c : guard_list)
    c->dump();
}

bool operator==(const guard2tc &g1, const guard2tc &g2)
{
  // Fast equality #1: same underlying node. Copy of the same guard or
  // a self-compare hits this — extremely cheap, two pointer reads.
  // Both nil also matches (two empty/true guards). The cached chain
  // is deterministic in guard_list, so shared base ⇒ matching list
  // under our mutator invariants.
  if (
    static_cast<const expr2tc &>(g1).get() ==
    static_cast<const expr2tc &>(g2).get())
    return true;

  // Fast inequality: if both guards have a cached crc and they differ,
  // the underlying and-chains differ, so the conjuncts must too. This
  // turns repeated comparisons of long guards (e.g. in symex state
  // dedup) into an O(1) atomic load instead of a vector walk. The crc
  // is computed lazily, so we only get the shortcut when something
  // else has already hashed the guard.
  if (!is_nil_expr(g1) && !is_nil_expr(g2))
  {
    size_t c1 = g1->crc_val.load(std::memory_order_acquire);
    size_t c2 = g2->crc_val.load(std::memory_order_acquire);
    if (c1 != 0 && c2 != 0 && c1 != c2)
      return false;
  }
  return g1.guard_list == g2.guard_list;
}

bool guard2tc::disjunction_may_simplify(const guard2tc &other) const
{
  if (is_true() || is_false() || other.is_true() || other.is_false())
    return true;

  expr2tc me = as_expr();
  expr2tc og = other.as_expr();
  if ((is_single_symbol() || is_and2t(me)) && is_and2t(og))
    return true;

  make_not(og);
  return me == og;
}

bool guard2tc::is_true() const
{
  return guard_list.empty();
}

bool guard2tc::is_false() const
{
  if (guard_list.size() != 1)
    return false;
  return guard_list.front() == gen_false_expr();
}

void guard2tc::make_true()
{
  clear();
}

void guard2tc::make_false()
{
  clear();
  guard_list.push_back(gen_false_expr());
  expr2tc::operator=(guard_list.front());
}

bool guard2tc::is_single_symbol() const
{
  return guard_list.size() == 1;
}

void guard2tc::clear()
{
  guard_list.clear();
  expr2tc::reset();
}
