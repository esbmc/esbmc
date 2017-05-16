/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/guard.h>
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
    g_expr = expr;
  }
  else
  {
    // Otherwise, just update the chain of ands
    and2tc new_g_expr(g_expr, expr);
    g_expr.swap(new_g_expr);
  }
}

void guardt::guard_expr(expr2tc& dest) const
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

  dest = expr2tc(new implies2t(as_expr(), dest));
}

void guardt::build_guard_expr()
{
  // This method closely related to guardt::add and guardt::guard_expr
  // We need to build the chain of ands, to avoid memory bloat on as_expr

  // if the guard is true, or a single symbol, we don't need to build it
  if(is_true() || is_single_symbol())
    return;

  // This method will only be used, when the guard is nil, for instance,
  // guardt &operator -= and guardt &operator |=, all other cases should
  // be handled by guardt::add
  assert(is_nil_expr(g_expr));

  // We can assume at least two operands
  auto it = guard_list.begin();

  expr2tc arg1, arg2;
  arg1 = *it++;
  arg2 = *it++;
  and2tc res(arg1, arg2);
  while (it != guard_list.end())
    res = and2tc(res, *it++);

  g_expr.swap(res);
}

void guardt::append(const guardt &guard)
{
  for(auto it : guard.guard_list)
    add(it);
}

guardt &operator -= (guardt &g1, const guardt &g2)
{
  guardt::guard_listt diff;
  std::set_difference(
    g1.guard_list.begin(),
    g1.guard_list.end(),
    g2.guard_list.begin(),
    g2.guard_list.end(),
    std::back_inserter(diff));

  // Clear g1 and build the guard's list and expr
  g1.clear();

  g1.guard_list.swap(diff);
  g1.build_guard_expr();

  return g1;
}

guardt &operator |= (guardt &g1, const guardt &g2)
{
  // Easy cases
  if(g2.is_false() || g1.is_true()) return g1;
  if(g1.is_false() || g2.is_true()) { g1 = g2; return g1; }

  if(g1.is_single_symbol() && g2.is_single_symbol())
  {
    // Both guards have one symbol, so check if we opposite symbols, e.g,
    // g1 == sym1 and g2 == !sym1
    expr2tc or_expr(new or2t(*g1.guard_list.begin(), *g2.guard_list.begin()));
    simplify(or_expr);

    if(::is_true(or_expr)) { g1.make_true(); return g1; }

    // Despite if we could simplify or not, clear and set the new guard
    g1.clear_insert(or_expr);
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

    // Common guards
    guardt common;
    std::set_intersection(
      g1.guard_list.begin(),
      g1.guard_list.end(),
      g2.guard_list.begin(),
      g2.guard_list.end(),
      std::back_inserter(common.guard_list));
    common.build_guard_expr();

    // New g1 and g2, without the common guards
    guardt new_g1;
    std::set_difference(
      g1.guard_list.begin(),
      g1.guard_list.end(),
      common.guard_list.begin(),
      common.guard_list.end(),
      std::back_inserter(new_g1.guard_list));
    new_g1.build_guard_expr();

    guardt new_g2;
    std::set_difference(
      g2.guard_list.begin(),
      g2.guard_list.end(),
      common.guard_list.begin(),
      common.guard_list.end(),
      std::back_inserter(new_g2.guard_list));
    new_g2.build_guard_expr();

    // Get the and expression from both guards
    expr2tc or_expr(new or2t(new_g1.as_expr(), new_g2.as_expr()));

    // If the guards single symbols, try to simplify the or expression
    if(new_g1.is_single_symbol() && new_g2.is_single_symbol())
      simplify(or_expr);

    g1.clear_append(common);
    g1.add(or_expr);
  }

  return g1;
}

void
guardt::dump(void) const
{
  for (auto it : guard_list)
    it->dump();
}

bool
operator == (const guardt &g1, const guardt &g2)
{
  // Very simple: the guard list should be identical.
  return g1.guard_list == g2.guard_list;
}

void guardt::swap(guardt& g)
{
  guard_list.swap(g.guard_list);
  g_expr.swap(g.g_expr);
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
  guard_list.clear();
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

void guardt::clear_append(const guardt& guard)
{
  clear();
  append(guard);
}

void guardt::clear_insert(const expr2tc& expr)
{
  clear();
  add(expr);
}

#ifdef WITH_PYTHON
#include <boost/python.hpp>
void
build_guard_python_class()
{
  using namespace boost::python;

  class_<guardt>("guardt")
    .def("add", &guardt::add)
    .def("append", &guardt::append)
    .def("as_expr", &guardt::as_expr)
    .def("guard_expr", &guardt::guard_expr)
    .def("is_true", &guardt::is_true)
    .def("is_false", &guardt::is_false)
    .def("make_true", &guardt::make_true)
    .def("make_false", &guardt::make_false)
    .def("swap", &guardt::swap)
    .def("dump", &guardt::dump)
    .def("clear", &guardt::clear);
}
#endif
