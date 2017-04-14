/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_VALUE_SET_DOMAIN_H
#define CPROVER_POINTER_ANALYSIS_VALUE_SET_DOMAIN_H

#include <goto-programs/static_analysis.h>
#include <pointer-analysis/value_set.h>
#include <util/irep2.h>
#include <util/migrate.h>

class value_set_domaint:public abstract_domain_baset
{
public:
  value_set_domaint() : value_set(NULL)
  {
  }

  ~value_set_domaint() {
    if (value_set)
      delete value_set;
    return;
  }

  value_set_domaint(const value_set_domaint &ref)
  {
    if (ref.value_set)
      value_set = new value_sett(*ref.value_set);
    else
      value_set = NULL;
  }

  value_sett *value_set;

  // overloading  

  virtual bool merge(const value_set_domaint &other, bool keepnew)
  {
    return value_set->make_union(*other.value_set, keepnew);
  }

  virtual void output(
    const namespacet &ns __attribute__((unused)),
    std::ostream &out) const
  {
    value_set->output(out);
  }
    
  virtual void initialize(
    const namespacet &ns,
    locationt l)
  {
    value_set = new value_sett(ns);
    value_set->clear();
    value_set->location_number=l->location_number;
  }

  virtual void transform(
    const namespacet &ns,
    locationt from_l,
    locationt to_l);

  virtual void get_reference_set(
    const namespacet &ns __attribute__((unused)),
    const expr2tc &expr,
    value_setst::valuest &dest)
  {
    value_set->get_reference_set(expr, dest);
  }
  
};

#endif
