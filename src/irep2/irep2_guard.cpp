#include <algorithm>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_utils.h>
#include <util/std_expr.h>

expr2tc guard2tc::as_expr() const
{
  // Mutators maintain the expr2tc base so it already matches the
  // current chain for size >= 1: single conjunct → that conjunct,
  // multi → and-chain. Only the empty-list case needs a translation
  // (base stays nil so default-constructed guards don't allocate).
  return is_true() ? gen_true_expr() : static_cast<const expr2tc &>(*this);
}

void guard2tc::add(const expr2tc &expr)
{
  if (is_false() || ::is_true(expr))
    return;

  if (is_true() || ::is_false(expr))
    clear();
  else if (is_and2t(expr))
  {
    const and2t &theand = to_and2t(expr);
    add(theand.side_1);
    add(theand.side_2);
    return;
  }

  guard_list.push_back(expr);

  // Update the chain of ands

  // Easy case, there is no chain yet
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
  for (const auto &c : other.guard_list)
    add(c);
}

guard2tc &operator-=(guard2tc &g1, const guard2tc &g2)
{
  std::vector<expr2tc> diff;
  std::set_difference(
    g1.guard_list.begin(),
    g1.guard_list.end(),
    g2.guard_list.begin(),
    g2.guard_list.end(),
    std::back_inserter(diff));

  g1.clear();
  g1.guard_list.swap(diff);
  g1.build_guard_expr();
  return g1;
}

guard2tc &operator|=(guard2tc &g1, const guard2tc &g2)
{
  if (g2.is_false() || g1.is_true())
    return g1;
  if (g1.is_false() || g2.is_true())
  {
    g1 = g2;
    return g1;
  }

  if (g1.is_single_symbol() && g2.is_single_symbol())
  {
    expr2tc or_expr = or2tc(g1.guard_list.front(), g2.guard_list.front());
    simplify(or_expr);

    if (::is_true(or_expr))
    {
      g1.make_true();
      return g1;
    }

    g1.clear_insert(or_expr);
    return g1;
  }

  // Factor out the common prefix: g1 || g2 == common && (g1' || g2').
  guard2tc common;
  std::set_intersection(
    g1.guard_list.begin(),
    g1.guard_list.end(),
    g2.guard_list.begin(),
    g2.guard_list.end(),
    std::back_inserter(common.guard_list));
  common.build_guard_expr();

  guard2tc new_g1;
  std::set_difference(
    g1.guard_list.begin(),
    g1.guard_list.end(),
    common.guard_list.begin(),
    common.guard_list.end(),
    std::back_inserter(new_g1.guard_list));
  new_g1.build_guard_expr();

  guard2tc new_g2;
  std::set_difference(
    g2.guard_list.begin(),
    g2.guard_list.end(),
    common.guard_list.begin(),
    common.guard_list.end(),
    std::back_inserter(new_g2.guard_list));
  new_g2.build_guard_expr();

  expr2tc or_expr = or2tc(new_g1.as_expr(), new_g2.as_expr());
  if (new_g1.is_single_symbol() && new_g2.is_single_symbol())
    simplify(or_expr);

  g1.clear_append(common);
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

void guard2tc::clear_append(const guard2tc &other)
{
  clear();
  append(other);
}

void guard2tc::clear_insert(const expr2tc &expr)
{
  clear();
  add(expr);
}
