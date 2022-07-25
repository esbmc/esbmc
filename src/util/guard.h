#ifndef CPROVER_GUARD_H
#define CPROVER_GUARD_H

#include <util/expr.h>
#include <irep2/irep2.h>
#include <util/migrate.h>

class guardt
{
public:
  // Default constructors
  guardt() = default;
  guardt(const guardt &ref) = default;

  typedef std::vector<expr2tc> guard_listt;

  void add(const expr2tc &expr);
  void append(const guardt &guard);

  expr2tc as_expr() const;
  void guard_expr(expr2tc &dest) const;

  bool disjunction_may_simplify(const guardt &other_guard) const;
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
