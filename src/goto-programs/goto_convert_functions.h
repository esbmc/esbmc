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

  /**
   * Ensure that the given type is complete, that is, has a known size. This is done by recursively following
   * symbol types and replacing them with the value of the referenced type.
   *
   * @param type the type to ensure is complete
   * @return true if the type was changed (i.e., it was incomplete and now is complete)
   * @return false if the type was already complete
   */
  bool ensure_type_is_complete(typet &type);
  /**
   * Ensure that all types in the context are complete which are not marked as incomplete.
   * Then replace all symbol types that are not pointers to symbols with the type they reference.
   */
  void thrash_type_symbols();
  /**
   * Visit the given irept and collect all symbol types that are not pointers to symbols.
   *
   * @param irept_val the irept to visit
   * @param to_replace the set of irept pointers to replace
   */
  void visit_irept(irept &irept_val, std::set<irept *> &to_replace);
  /**
   * Called for every type and subtype of a symbol. If the type is a symbol, it is added to the set of
   * types to replace.
   *
   * @param type the type to visit
   * @param to_replace the set of irept pointers to replace
   */
  void visit_sub_type(irept &type, std::set<irept *> &to_replace);

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
