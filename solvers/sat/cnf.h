/*******************************************************************\

Module: CNF Generation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_PROP_CNF_H
#define CPROVER_PROP_CNF_H

#include <list>

#include <solvers/prop/prop.h>

class cnft:public virtual propt
{
public:
  cnft();
  virtual ~cnft();

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
  virtual unsigned no_clauses() const=0;

  static void eliminate_duplicates(const bvt &bv, bvt &dest);

protected:
  unsigned _no_variables;
  
  bool process_clause(const bvt &bv, bvt &dest);

  static bool is_all(const bvt &bv, literalt l)
  {
    for(unsigned i=0; i<bv.size(); i++)
      if(bv[i]!=l) return false;
    return true;
  }
};

class cnf_solvert:public cnft
{
public:
  cnf_solvert():status(INIT), clause_counter(0)
  {
  }
  
  virtual unsigned no_clauses() const
  {
    return clause_counter;
  }

protected:
  typedef enum { INIT, SAT, UNSAT, ERROR } statust;
  statust status;
  unsigned clause_counter;
}; 

#endif
