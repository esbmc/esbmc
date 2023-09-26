#include <algorithm>
#include <util/guard.h>
#include <irep2/irep2_utils.h>
#include <util/std_expr.h>

expr2tc guardt::as_expr() const
{
  if(is_true())
    return gen_true_expr();

  if(is_single_symbol())
    return *guard_list.begin();

  assert(!is_nil_expr(g_expr));
  return g_expr;
}

void guardt::add(const expr2tc &expr)
{
  if(is_false() || ::is_true(expr))
    return;

  if(is_true() || ::is_false(expr))
  {
    clear();
  }
  else if(is_and2t(expr))
  {
    const and2t &theand = to_and2t(expr);
    add(theand.side_1);
    add(theand.side_2);
    return;
  }

  guard_list.push_back(expr);

  // Update the chain of ands

  // Easy case, there is no g_expr
  if(is_nil_expr(g_expr))
  {
    assert(guard_list.size() == 1);
    g_expr = expr;
  }
  else
  {
    // Otherwise, just update the chain of ands
    expr2tc new_g_expr = and2tc(g_expr, expr);
    g_expr.swap(new_g_expr);
  }
}

void guardt::guard_expr(expr2tc &dest) const
{
  // Fills the expr only if it's not true
  if(is_true())
    return;

  if(::is_false(dest))
  {
    dest = as_expr();
    make_not(dest);
    return;
  }

  dest = implies2tc(as_expr(), dest);
}

void guardt::build_guard_expr()
{
  // This method closely related to guardt::add and guardt::guard_expr
  // We need to build the chain of ands, to avoid memory bloat on as_expr

  // This method will only be used, when the guard is nil, for instance,
  // guardt &operator -= and guardt &operator |=, all other cases should
  // be handled by guardt::add
  assert(is_nil_expr(g_expr));

  // if the guard is true, we don't need to build it
  if(is_true())
    return;

  // the expression is the single symbol in case the list just has this one
  if(is_single_symbol())
  {
    g_expr = *guard_list.begin();
    return;
  }

  // We can assume at least two operands
  auto it = guard_list.begin();

  expr2tc arg1, arg2;
  arg1 = *it++;
  arg2 = *it++;
  expr2tc res = and2tc(arg1, arg2);
  while(it != guard_list.end())
    res = and2tc(res, *it++);

  g_expr.swap(res);
}

void guardt::append(const guardt &guard)
{
  for(auto const &it : guard.guard_list)
    add(it);
}

guardt &operator-=(guardt &g1, const guardt &g2)
{
  std::unordered_set<expr2tc, irep2_hash> s2(
    g2.guard_list.begin(), g2.guard_list.end());
  expr2tc *e = g1.guard_list.data();
  size_t n = g1.guard_list.size();
  const expr2tc *end = e + n;
  for(expr2tc *f = e; f < end; f++)
  {
    if(s2.find(*f) == s2.end())
      *e++ = *f;
  }
  size_t m = e - g1.guard_list.data();
  if(n != m)
  {
    g1.guard_list.resize(m);
    g1.g_expr.reset();
    g1.build_guard_expr();
  }

  return g1;
}

guardt::guardt(guard_listt guard_list) noexcept
  : guard_list(std::move(guard_list))
{
  build_guard_expr();
}

guardt &operator|=(guardt &g1, const guardt &g2)
{
  // Easy cases
  if(g2.is_false() || g1.is_true())
    return g1;
  if(g1.is_false() || g2.is_true())
  {
    g1 = g2;
    return g1;
  }

  if(g1.is_single_symbol() && g2.is_single_symbol())
  {
    // Both guards have one symbol, so check if we have opposite symbols, e.g,
    // g1 == sym1 and g2 == !sym1
    expr2tc or_expr = or2tc(*g1.guard_list.begin(), *g2.guard_list.begin());
    simplify(or_expr);

    if(::is_true(or_expr))
    {
      g1.make_true();
      return g1;
    }

    // Despite if we could simplify or not, clear and set the new guard
    g1.clear();
    g1.add(or_expr);
  }
  else
  {
    // Here, we have a symbol (or symbols) in g2 to be or'd with the symbol
    // (or symbols) in g1, e.g:
    // g1 = !guard3 && !guard2 && !guard1
    // g2 = guard2 && !guard1
    // res = g1 || g2 = (!guard3 && !guard2 && !guard1) || (guard2 && !guard1)

    // Simplify equation: everything that's common in both guards, will not
    // be or'd

    std::unordered_set<expr2tc, irep2_hash> s1(
      g1.guard_list.begin(), g1.guard_list.end());
    std::unordered_set<expr2tc, irep2_hash> s2(
      g2.guard_list.begin(), g2.guard_list.end());
    guardt common, n1, n2;
    for(const expr2tc &e : g1.guard_list)
    {
      guardt &g = s2.find(e) != s2.end() ? common : n1;
      g.add(e);
    }
    for(const expr2tc &e : g2.guard_list)
      if(s1.find(e) == s1.end())
        n2.add(e);

    // Get the and expression from both guards
    expr2tc or_expr = or2tc(n1.as_expr(), n2.as_expr());

    // If the guards are single symbols, try to simplify the or expression
    if(n1.is_single_symbol() && n2.is_single_symbol())
      simplify(or_expr);

    // keep common guards and add the new or_expr
    g1 = std::move(common);
    g1.add(or_expr);
  }

  return g1;
}

void guardt::dump() const
{
  for(auto const &it : guard_list)
    it->dump();
}

bool operator==(const guardt &g1, const guardt &g2)
{
  // Very simple: the guard list should be identical.
  return g1.guard_list == g2.guard_list;
}

void guardt::swap(guardt &g)
{
  guard_list.swap(g.guard_list);
  g_expr.swap(g.g_expr);
}

bool guardt::disjunction_may_simplify(const guardt &other_guard) const
{
  if(is_true() || is_false() || other_guard.is_true() || other_guard.is_false())
    return true;

  auto og_expr = other_guard.as_expr();
  if((is_single_symbol() || is_and2t(as_expr())) && is_and2t(og_expr))
    return true;

  make_not(og_expr);
  if(as_expr() == og_expr)
    return true;

  return false;
}

bool guardt::is_true() const
{
  return guard_list.empty();
}

bool guardt::is_false() const
{
  // Never false
  if(guard_list.size() != 1)
    return false;

  return (*guard_list.begin() == gen_false_expr());
}

void guardt::make_true()
{
  clear();
}

void guardt::make_false()
{
  add(gen_false_expr());
}

bool guardt::is_single_symbol() const
{
  return (guard_list.size() == 1);
}

void guardt::clear()
{
  guard_list.clear();
  g_expr.reset();
}
