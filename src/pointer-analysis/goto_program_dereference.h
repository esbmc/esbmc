/*******************************************************************\

Module: Value Set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_POINTER_ANALYSIS_GOTO_PROGRAM_DEREFERENCE_H
#define CPROVER_POINTER_ANALYSIS_GOTO_PROGRAM_DEREFERENCE_H

#include <goto-programs/goto_functions.h>
#include <pointer-analysis/dereference.h>
#include <pointer-analysis/value_sets.h>
#include <util/namespace.h>

class goto_program_dereferencet:protected dereference_callbackt
{
public:
  goto_program_dereferencet(
    const namespacet &_ns,
    contextt &_new_context,
    const optionst &_options,
    value_setst &_value_sets):
    options(_options),
    ns(_ns),
    value_sets(_value_sets),
    dereference(_ns, _new_context, _options, *this) { }

  void dereference_program(
    goto_programt &goto_program,
    bool checks_only=false);

  void dereference_program(
    goto_functionst &goto_functions,
    bool checks_only=false);

  void pointer_checks(goto_programt &goto_program);
  void pointer_checks(goto_functionst &goto_functions);

  void dereference_expression(
    goto_programt::const_targett target,
    expr2tc &expr);

  virtual ~goto_program_dereferencet()
  {
  }

protected:
  const optionst &options;
  const namespacet &ns;
  value_setst &value_sets;
  dereferencet dereference;

  virtual bool is_valid_object(const irep_idt &identifier);

  virtual bool has_failed_symbol(
    const expr2tc &expr,
    const symbolt *&symbol);

  virtual void dereference_failure(
    const std::string &property,
    const std::string &msg,
    const guardt &guard);

  virtual void get_value_set(const expr2tc &expr, value_setst::valuest &dest);

  void dereference_instruction(
    goto_programt::targett target,
    bool checks_only=false);

protected:
  void dereference_expr(expr2tc &expr, const bool checks_only,
                        const dereferencet::modet mode);

  goto_programt::local_variablest *valid_local_variables;
  locationt dereference_location;
  goto_programt::const_targett current_target;

  std::set<expr2tc> assertions;
  goto_programt new_code;
};

void dereference(
  goto_programt::const_targett target,
  expr2tc &expr,
  const namespacet &ns,
  value_setst &value_sets);

void remove_pointers(
  goto_programt &goto_program,
  contextt &context,
  const optionst &options,
  value_setst &value_sets);

void remove_pointers(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  value_setst &value_sets);

void pointer_checks(
  goto_programt &goto_program,
  const namespacet &ns,
  const optionst &options,
  value_setst &value_sets);

void pointer_checks(
  goto_functionst &goto_functions,
  const namespacet &ns,
  contextt &context,
  const optionst &options,
  value_setst &value_sets);

#endif
