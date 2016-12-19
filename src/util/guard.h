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

  typedef std::list<expr2tc> guard_listt;

  void add(const expr2tc &expr);
  void move(expr2tc &expr);

  void append(const guardt &guard)
  {
    for(guard_listt::const_iterator it=guard.guard_list.begin();
        it!=guard.guard_list.end();
        it++)
      add(*it);
  }

  expr2tc as_expr(guard_listt::const_iterator it) const;

  expr2tc as_expr() const
  {
    return as_expr(guard_list.begin());
  }
  
  void guard_expr(expr2tc &dest) const
  {
    if(guard_list.empty())
    {
    }
    else
    {
      dest = expr2tc(new implies2t(as_expr(), dest));
    }
  }

  bool empty() const { return guard_list.empty(); }
  bool is_true() const { return empty(); } 
  bool is_false() const;
  
  void make_true()
  {
    guard_list.clear();
  }
  
  void make_false()
  {
    guard_list.clear();
    expr2tc tmp = false_expr;
    guard_list.push_back(tmp);
  }
  
  friend guardt &operator -= (guardt &g1, const guardt &g2);
  friend guardt &operator |= (guardt &g1, const guardt &g2);
  friend bool operator == (const guardt &g1, const guardt &g2);

  void back_sub(const guardt &g2);
  
  void swap(guardt &g)
  {
    guard_list.swap(g.guard_list);
  }

  friend std::ostream &operator << (std::ostream &out, const guardt &g);

  void dump() const;
  
  unsigned size() const
  {
    return guard_list.size();
  }
  
  void resize(unsigned s)
  {
    guard_list.resize(s);
  }
  
  const guard_listt &get_guard_list() const
  {
    return guard_list;
  }

protected:
  guard_listt guard_list;  
};

#define Forall_guard(it, guard_list) \
  for(guardt::guard_listt::iterator it=(guard_list).begin(); \
      it!=(guard_list).end(); it++)

#define forall_guard(it, guard_list) \
  for(guardt::guard_listt::const_iterator it=(guard_list).begin(); \
      it!=(guard_list).end(); it++)

#endif
