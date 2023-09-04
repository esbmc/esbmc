#ifndef CPROVER_GOTO_CONVERT_FUNCTIONS_H
#define CPROVER_GOTO_CONVERT_FUNCTIONS_H

#include <goto-programs/goto_convert_class.h>
#include <goto-programs/goto_functions.h>

// just convert it all
void goto_convert(
  contextt &context,
  optionst &options,
  goto_functionst &functions);

class goto_convert_functionst : public goto_convertt
{
public:
  typedef std::map<irep_idt, std::set<irep_idt>> typename_mapt;
  typedef std::set<irep_idt> typename_sett;

  void goto_convert();
  void convert_function(symbolt &symbol);
  void thrash_type_symbols();
  void rename_types(
    irept &type,
    const symbolt &cur_name_sym,
    const std::unordered_set<irep_idt, irep_id_hash> &avoid,
    bool under_ptr);
  void rename_exprs(
    irept &expr,
    const symbolt &cur_name_sym,
    const std::unordered_set<irep_idt, irep_id_hash> &avoid,
    bool under_ptr);

  goto_convert_functionst(
    contextt &_context,
    optionst &_options,
    goto_functionst &_functions);

protected:
  goto_functionst &functions;

  static bool hide(const goto_programt &goto_program);

  //
  // function calls
  //
  void add_return(goto_functiont &f, const locationt &location);
};

#endif
