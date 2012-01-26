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

class z3_propt:virtual public propt
{
public:
  z3_propt(bool uw);
  virtual ~z3_propt();

//  virtual literalt constant(bool value)
//  { return value?l_const1:l_const0; }
#if 1
  virtual void land(literalt a, literalt b, literalt o);
  virtual void lor(literalt a, literalt b, literalt o);
  virtual void lxor(literalt a, literalt b, literalt o);
  virtual void lnand(literalt a, literalt b, literalt o);
  virtual void lnor(literalt a, literalt b, literalt o);
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
  virtual literalt lnand(literalt a, literalt b);
  virtual literalt lnor(literalt a, literalt b);
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
  //virtual void set_assignment(literalt a, bool value);
  virtual propt::resultt prop_solve();

  friend class z3_convt;
  friend class z3_dect;

  virtual void clear()
  {
    assignment.clear();
  }

  void reset_assignment()
  {
    assignment.clear();
    assignment.resize(no_variables(), tvt(tvt::TV_UNKNOWN));
  }

private:
	z3_capi z3_api;
	Z3_context z3_ctx;
	typedef std::map<std::string, Z3_ast> map_prop_varst;
	map_prop_varst map_prop_vars;
	std::list<Z3_ast> assumpt;
    bool store_assumptions;
    bool smtlib;

protected:
  unsigned _no_variables;
  std::ostream &out;
  bool uw; // Are we doing underapprox+widenning?
           // Affects how formula are constructed

  Z3_ast z3_literal(literalt l);

  std::vector<tvt> assignment;

  bool process_clause(const bvt &bv, bvt &dest);
#if 0
  static bool is_all(const bvt &bv, literalt l)
  {
    for(unsigned i=0; i<bv.size(); i++)
      if(bv[i]!=l) return false;
    return true;
  }
#endif

  void assert_formula(Z3_ast ast, bool needs_literal = true);
  void assert_literal(literalt l, Z3_ast formula);
};

#endif
