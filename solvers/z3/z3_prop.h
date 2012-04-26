/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_Z3_PROP_H
#define CPROVER_PROP_Z3_PROP_H

#include <iostream>
#include <solvers/prop/prop.h>

#include "z3_capi.h"

typedef unsigned int uint;
class z3_convt; // forward dec

class z3_propt:virtual public propt
{
public:
  z3_propt(bool uw, z3_convt &_owner);
  virtual ~z3_propt();

//  virtual literalt constant(bool value)
//  { return value?l_const1:l_const0; }
#if 1
  virtual void land(literalt a, literalt b, literalt o);
  virtual void lor(literalt a, literalt b, literalt o);
  virtual void lxor(literalt a, literalt b, literalt o);
  virtual void lequal(literalt a, literalt b, literalt o);
  virtual void limplies(literalt a, literalt b, literalt o);
#endif
  virtual literalt land(literalt a, literalt b);
  virtual literalt lor(literalt a, literalt b);
  virtual literalt land(const bvt &bv);
  virtual literalt lor(const bvt &bv);
  virtual literalt lxor(const bvt &bv);
  virtual literalt lnot(literalt a);
  virtual literalt lxor(literalt a, literalt b);
  virtual literalt lequal(literalt a, literalt b);
  virtual literalt limplies(literalt a, literalt b);
  virtual literalt lselect(literalt a, literalt b, literalt c); // a?b:c
  virtual literalt new_variable();
  virtual unsigned no_variables() const { return _no_variables; }
  virtual void set_no_variables(unsigned no) { _no_variables=no; }
  virtual void lcnf(const bvt &bv);

  static void eliminate_duplicates(const bvt &bv, bvt &dest);

  virtual const std::string solver_text()
  { return "Z3"; }

  virtual tvt l_get(literalt a) const;
  virtual propt::resultt prop_solve();

  friend class z3_convt;

private:
  z3_capi z3_api;
  Z3_context z3_ctx;
  std::list<Z3_ast> assumpt;
  bool store_assumptions;
  bool smtlib;

  z3_convt &owner; // Reference back to convt owner.

protected:
  unsigned _no_variables;
  bool uw; // Are we doing underapprox+widenning?
           // Affects how formula are constructed

  Z3_ast z3_literal(literalt l);

  bool process_clause(const bvt &bv, bvt &dest);
  void assert_formula(Z3_ast ast, bool needs_literal = true);
  void assert_literal(literalt l, Z3_ast formula);
};

#endif
