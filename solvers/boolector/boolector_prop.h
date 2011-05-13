/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#ifndef CPROVER_PROP_BOOLECTOR_PROP_H
#define CPROVER_PROP_BOOLECTOR_PROP_H

#include <iostream>
#include <solvers/prop/prop.h>

struct BtorExp;
struct Btor;

class boolector_propt:virtual public propt
{
public:
  boolector_propt(std::ostream &_out);

  virtual ~boolector_propt();

//  virtual literalt constant(bool value)
//  { return value?l_const1:l_const0; }

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
  virtual void set_no_variables(unsigned no) { _no_variables=no; }

  virtual void lcnf(const bvt &bv);

  static void eliminate_duplicates(const bvt &bv, bvt &dest);
  BtorExp* convert_literal(unsigned l);

  virtual const std::string solver_text()
  { return "Boolector"; }

  virtual tvt l_get(literalt a) const;

  virtual propt::resultt prop_solve();

  friend class boolector_convt;
  friend class boolector_dect;

  virtual void clear()
  {
    assignment.clear();
  }

  void reset_assignment()
  {
    assignment.clear();
    assignment.resize(no_variables(), tvt(tvt::TV_UNKNOWN));
  }

	Btor *boolector_ctx;

private:
  std::vector<BtorExp*> assumpt;
  bool btor;

protected:
  unsigned _no_variables;
  std::ostream &out;

  BtorExp* boolector_literal(literalt l);
  typedef hash_map_cont<unsigned, BtorExp*> literal_cachet;
  literal_cachet literal_cache;

  std::vector<tvt> assignment;

  bool process_clause(const bvt &bv, bvt &dest);

  static bool is_all(const bvt &bv, literalt l)
  {
    for(unsigned i=0; i<bv.size(); i++)
      if(bv[i]!=l) return false;
    return true;
  }

};

#endif
