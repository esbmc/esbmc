/*******************************************************************\

Module: Guard Data Structure

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GUARD_H
#define CPROVER_GUARD_H

#include <irep2.h>
#include <migrate.h>

#include <iostream>

#include <expr.h>

class guardt
{
public:
  // Default constructors
  guardt() : guard_list() { }
  guardt(const guardt &ref) { *this = ref; }

  typedef std::vector<expr2tc> guard_listt;

  void add(const expr2tc &expr);
  void append(const guardt &guard);
  void clear_insert(const expr2tc &expr);
  void clear_append(const guardt &guard);

  expr2tc as_expr() const;
  void guard_expr(expr2tc &dest) const;

  bool empty() const;
  bool is_true() const;
  bool is_false() const;
  bool is_single_symbol() const;

  void make_true();
  void make_false();
  void swap(guardt &g);

  friend guardt &operator -= (guardt &g1, const guardt &g2);
  friend guardt &operator |= (guardt &g1, const guardt &g2);
  friend bool operator == (const guardt &g1, const guardt &g2);

  guard_listt::size_type size() const;
  void resize(guard_listt::size_type size);

  void dump() const;

protected:
  guard_listt guard_list;
};

#endif
