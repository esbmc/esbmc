#ifndef CPROVER_POINTER_ANALYSIS_VALUE_PROPAGATION_H
#define CPROVER_POINTER_ANALYSIS_VALUE_PROPAGATION_H

#include <goto-programs/static_analysis.h>
#include <pointer-analysis/value_set_domain.h>
#include <pointer-analysis/value_sets.h>
#include <util/base/xml.h>

class value_set_analysist : public value_setst,
                            public static_analysist<value_set_domaint>
{
public:
  explicit value_set_analysist(const namespacet &_ns)
    : static_analysist<value_set_domaint>(_ns)
  {
  }

  typedef static_analysist<value_set_domaint> baset;

  // value_setst and static_analysis_baset each declare a locationt (both
  // goto_programt::const_targett); name one to keep member declarations
  // unambiguous.
  using locationt = baset::locationt;

  // overloading
  void initialize(const goto_programt &goto_program) override;
  void initialize(const goto_functionst &goto_functions) override;

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

  void get_globals(std::list<value_sett::entryt> &dest);

  void get_entries_rec(
    const std::string &identifier,
    const std::string &suffix,
    const typet &type,
    std::list<value_sett::entryt> &dest);

protected:
  bool check_type(const typet &type);
  void add_vars(const goto_functionst &goto_functions);
  void add_vars(const goto_programt &goto_programa);

  void get_entries(const symbolt &symbol, std::list<value_sett::entryt> &dest);

public:
  // interface value_sets
  void get_values(locationt l, const expr2tc &expr, value_setst::valuest &dest)
    override
  {
    (*this)[l].value_set->get_value_set(expr, dest);
  }

  // Both members already exist on the static_analysist base, but a member
  // inherited from one base does not override a pure virtual declared in
  // another, so value_setst needs these explicit forwarders.
  void get_reference_set(
    locationt l,
    const expr2tc &expr,
    value_setst::valuest &dest) override
  {
    baset::get_reference_set(l, expr, dest);
  }

  bool has_location(locationt l) const override
  {
    return baset::has_location(l);
  }
};

#endif
