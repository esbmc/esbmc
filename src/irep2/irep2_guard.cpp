#include <algorithm>
#include <unordered_set>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_utils.h>

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

  g1.clear();
  g1.guard_list.swap(diff);
  g1.build_guard_expr();
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

  // g1 || g2 == common && (g1' || g2'), where
  //   common = g1 ∩ g2  (as conjunct sets),
  //   g1'    = g1 \ common,
  //   g2'    = g2 \ common.
  // Hash-set membership rather than std::set_* because guard_list is
  // in insertion order, not sorted. Build g2_set once, then mutate it
  // as we scan g1: `erase(c)` returns 1 iff c is in common, in which
  // case the matched entry is removed from g2_set so what remains is
  // exactly g2's residual set. One set construction covers all three
  // outputs (common, new_g1, new_g2).
  std::unordered_set<expr2tc, irep2_hash> g2_set(
    g2.guard_list.begin(), g2.guard_list.end());

  guard2tc common;
  guard2tc new_g1;
  common.guard_list.reserve(std::min(g1.guard_list.size(), g2_set.size()));
  new_g1.guard_list.reserve(g1.guard_list.size());
  for (const auto &c : g1.guard_list)
  {
    if (g2_set.erase(c))
      common.guard_list.push_back(c);
    else
      new_g1.guard_list.push_back(c);
  }
  common.build_guard_expr();
  new_g1.build_guard_expr();

  guard2tc new_g2;
  new_g2.guard_list.reserve(g2_set.size());
  for (const auto &c : g2.guard_list)
    if (g2_set.count(c))
      new_g2.guard_list.push_back(c);
  new_g2.build_guard_expr();

  // If either residual is empty, that side equals `common` itself, so
  // the disjunction reduces to `common || (common && other) ≡ common`.
  // Skip the or2tc construction and chain-extend entirely. Covers the
  // pathological case where one guard is a prefix of the other (a
  // common pattern at branch joins where one side adds extra
  // conjuncts), and the identical-set case.
  if (new_g1.is_true() || new_g2.is_true())
  {
    g1 = std::move(common);
    return g1;
  }

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
