/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

//
// convert expression to boolean formula
//

#ifndef CPROVER_PROP_CONV_H
#define CPROVER_PROP_CONV_H

#include <string>
#include <list>

#include <hash_cont.h>
#include <decision_procedure.h>
#include <threeval.h>

#include "prop.h"

class prop_convt:virtual public decision_proceduret
{
public:
  explicit prop_convt(propt &_prop):
    use_cache(true),
    equality_propagation(true),
    prop(_prop) { }
  virtual ~prop_convt() { }

  // overloading
  virtual void set_to(const exprt &expr, bool value);
  virtual decision_proceduret::resultt dec_solve();

  virtual const std::string decision_procedure_text()
  { return "propositional reduction"; }

  // conversion with cache
  virtual literalt convert(const exprt &expr)
  { return convert(expr, use_cache); }

  // conversion without cache
  virtual literalt convert_nocache(const exprt &expr)
  { return convert(expr, false); }

  // get a boolean value from counterexample if not valid
  virtual bool get_bool(const exprt &expr, tvt &value) const;

  // get a value from counterexample if not valid
  virtual exprt get(const exprt &expr) const;
  
  // dump satisfying assignment
  virtual void print_assignment(std::ostream &out) const;
  
  // get literal for expression, if available
  virtual bool literal(const exprt &expr, literalt &literal) const;
  
  bool use_cache;
  bool equality_propagation;
  
  propt &prop;
  
  friend class prop_conv_store_constraintt;

  virtual void post_process();
  
  virtual void clear_cache()
  {
    cache.clear();
  }
  
protected:
  virtual literalt convert(const exprt &expr, bool do_cache);
  virtual literalt convert_rest(const exprt &expr);
  virtual literalt convert_bool(const exprt &expr);
  
  virtual bool set_equality_to_true(const exprt &expr);

  // symbols
  typedef std::map<irep_idt, literalt> symbolst;
  symbolst symbols;

  virtual literalt get_literal(const irep_idt &symbol);

  // cache
  typedef hash_map_cont<exprt, literalt, irep_hash> cachet;
  cachet cache;
  
  virtual void ignoring(const exprt &expr);
  
public:
  const cachet &get_cache() const { return cache; }
};

#endif
