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
  value_set_domaint() : value_set(nullptr)
  {
  }

  ~value_set_domaint() override {
    if (value_set)
      delete value_set;
  }

  value_set_domaint(const value_set_domaint &ref)
  {
    if (ref.value_set)
      value_set = new value_sett(*ref.value_set);
    else
      value_set = nullptr;
  }

  value_sett *value_set;

  // overloading  

  virtual bool merge(const value_set_domaint &other, bool keepnew)
  {
    return value_set->make_union(*other.value_set, keepnew);
  }

  void output(
    const namespacet &ns __attribute__((unused)),
    std::ostream &out) const override
  {
    value_set->output(out);
  }
    
  void initialize(
    const namespacet &ns,
    locationt l) override 
  {
    value_set = new value_sett(ns);
    value_set->clear();
    value_set->location_number=l->location_number;
  }

  void transform(
    const namespacet &ns,
    locationt from_l,
    locationt to_l) override ;

  void get_reference_set(
    const namespacet &ns __attribute__((unused)),
    const expr2tc &expr,
    value_setst::valuest &dest) override
  {
    value_set->get_reference_set(expr, dest);
  }
  
};

#endif
