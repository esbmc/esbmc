/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROP_SMT_PROP_H
#define CPROVER_PROP_SMT_PROP_H

#include <iostream>

#include <solvers/prop/prop.h>

class smt_propt:virtual public propt
{
public:
  smt_propt(std::ostream &_out);
  virtual ~smt_propt() { }

  virtual void land(literalt a, literalt b, literalt o);
  virtual void lor(literalt a, literalt b, literalt o);
  virtual void lxor(literalt a, literalt b, literalt o);
  virtual void lnand(literalt a, literalt b, literalt o);
  virtual void lnor(literalt a, literalt b, literalt o);
  virtual void lequal(literalt a, literalt b, literalt o);
  virtual void limplies(literalt a, literalt b, literalt o);

  virtual literalt land(literalt a, literalt b);
  virtual literalt lor(literalt a, literalt b);
  virtual literalt land(const bvt &bv);
  virtual literalt lor(const bvt &bv);
  virtual literalt lxor(const bvt &bv);
  virtual literalt lnot(literalt a);
  virtual literalt lxor(literalt a, literalt b);
  virtual literalt lnand(literalt a, literalt b);
  virtual literalt lnor(literalt a, literalt b);
  virtual literalt lequal(literalt a, literalt b);
  virtual literalt limplies(literalt a, literalt b);
  virtual literalt lselect(literalt a, literalt b, literalt c); // a?b:c
  virtual literalt new_variable();
  virtual unsigned no_variables() const { return _no_variables; }
  virtual void set_no_variables(unsigned no) { assert(false); }
  //virtual unsigned no_clauses()=0;

  virtual void lcnf(const bvt &bv);

  virtual const std::string solver_text()
  { return "SMT"; }

  virtual tvt l_get(literalt literal) const;

  virtual propt::resultt prop_solve();

  friend class smt_convt;
  friend class smt_dect;

  virtual void clear()
  {
    assignment.clear();
  }

  void reset_assignment()
  {
    assignment.clear();
    assignment.resize(no_variables(), tvt(tvt::TV_UNKNOWN));
  }

protected:
  unsigned _no_variables;
  std::ostream &out;

  std::string smt_literal(literalt l);
  literalt def_smt_literal();

  std::vector<tvt> assignment;
};

#endif
