/*******************************************************************\

Module: Value Set Propagation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_VALUE_PROPAGATION_H
#define CPROVER_POINTER_ANALYSIS_VALUE_PROPAGATION_H

#include <goto-programs/static_analysis.h>
#include <pointer-analysis/value_set_domain.h>
#include <pointer-analysis/value_sets.h>
#include <util/xml.h>

class value_set_analysist:
  public value_setst,
  public static_analysist<value_set_domaint>
{
public:
   value_set_analysist(const namespacet &_ns):
     static_analysist<value_set_domaint>(_ns)
   {
   }

  typedef static_analysist<value_set_domaint> baset;

  // overloading  
  void initialize(const goto_programt &goto_program) override ;
  void initialize(const goto_functionst &goto_functions) override ;

  friend void convert(
    const goto_functionst &goto_functions,
    const value_set_analysist &value_set_analysis,
    xmlt &dest);

  friend void convert(
    const goto_programt &goto_program,
    const value_set_analysist &value_set_analysis,
    xmlt &dest);

  void convert(
    const goto_programt &goto_program,
    const irep_idt &identifier,
    xmlt &dest) const;
    
protected:
  bool check_type(const typet &type);
  void get_globals(std::list<value_sett::entryt> &dest);
  void add_vars(const goto_functionst &goto_functions);
  void add_vars(const goto_programt &goto_programa);

  void get_entries(
    const symbolt &symbol,
    std::list<value_sett::entryt> &dest);

  void get_entries_rec(
    const std::string &identifier,
    const std::string &suffix,
    const typet &type,
    std::list<value_sett::entryt> &dest);

public:
  // interface value_sets
  void get_values(
    locationt l,
    const expr2tc &expr,
    value_setst::valuest &dest) override 
  {
    (*this)[l].value_set->get_value_set(expr, dest);
  }  
};

#endif
