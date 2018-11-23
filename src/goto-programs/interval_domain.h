/*******************************************************************\

Module: Interval Domain

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

/// \file
/// Interval Domain

#ifndef CPROVER_ANALYSES_INTERVAL_DOMAIN_H
#define CPROVER_ANALYSES_INTERVAL_DOMAIN_H

#include <goto-programs/ai.h>
#include <goto-programs/interval_template.h>
#include <util/ieee_float.h>
#include <util/irep2_utils.h>
#include <util/mp_arith.h>

typedef interval_templatet<mp_integer> integer_intervalt;

class interval_domaint : public ai_domain_baset
{
public:
  // Trivial, conjunctive interval domain for both float
  // and integers. The categorization 'float' and 'integers'
  // is done by is_int and is_float.

  interval_domaint() : bottom(true)
  {
  }

  void transform(
    goto_programt::const_targett from,
    goto_programt::const_targett to,
    ai_baset &ai,
    const namespacet &ns) final override;

  void output(std::ostream &out) const override;

  void dump() const
  {
    output(std::cout);
  }

protected:
  bool join(const interval_domaint &b);

public:
  bool merge(
    const interval_domaint &b,
    goto_programt::const_targett,
    goto_programt::const_targett)
  {
    return join(b);
  }

  // no states
  void make_bottom() final override
  {
    int_map.clear();
    bottom = true;
  }

  // all states
  void make_top() final override
  {
    int_map.clear();
    bottom = false;
  }

  void make_entry() final override
  {
    make_top();
  }

  bool is_bottom() const override final
  {
#if 0
    // This invariant should hold but is not correctly enforced at the moment.
    assert(!bottom || (int_map.empty() && float_map.empty()));
#endif

    return bottom;
  }

  bool is_top() const override final
  {
    return !bottom && int_map.empty();
  }

  expr2tc make_expression(const expr2tc &expr) const;

  void assume(const expr2tc &);

  virtual bool
  ai_simplify(expr2tc &condition, const namespacet &ns) const override;

protected:
  bool bottom;

  typedef hash_map_cont<irep_idt, integer_intervalt, irep_id_hash> int_mapt;

  int_mapt int_map;

  void havoc_rec(const expr2tc &expr);
  void assume_rec(const expr2tc &expr, bool negation = false);
  void assume_rec(const expr2tc &lhs, expr2t::expr_ids id, const expr2tc &rhs);
  void assign(const expr2tc &assignment);
  integer_intervalt get_int_rec(const expr2tc &expr);
};

#endif // CPROVER_ANALYSES_INTERVAL_DOMAIN_H
