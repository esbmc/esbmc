/*******************************************************************\

Module: Guard Data Structure

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GUARD_H
#define CPROVER_GUARD_H

#include <iostream>
#include <util/expr.h>
#include <util/irep2.h>
#include <util/migrate.h>

class guardt
{
  friend void build_guard_python_class();

public:
  // Default constructors
  guardt() = default;
  guardt(const guardt &ref)
  {
    *this = ref;
  }

  typedef std::vector<expr2tc> guard_listt;

  void add(const expr2tc &expr);
  void append(const guardt &guard);

  expr2tc as_expr() const;
  void guard_expr(expr2tc &dest) const;

  bool is_true() const;
  bool is_false() const;

  void make_true();
  void make_false();
  void swap(guardt &g);

  friend guardt &operator-=(guardt &g1, const guardt &g2);
  friend guardt &operator|=(guardt &g1, const guardt &g2);
  friend bool operator==(const guardt &g1, const guardt &g2);

  void dump() const;

protected:
  guard_listt guard_list;
  expr2tc g_expr;

  bool is_single_symbol() const;
  void clear();
  void clear_append(const guardt &guard);
  void clear_insert(const expr2tc &expr);

  void build_guard_expr();
};

#endif
