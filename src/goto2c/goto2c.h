/*******************************************************************\

Module:

Author: 

\*******************************************************************/

#ifndef CPROVER_GOTO2C_H
#define CPROVER_GOTO2C_H

#include <util/namespace.h>
#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_program.h>

class goto2ct
{
public:
  goto2ct(namespacet &_ns, goto_functionst _goto_functions)
    : ns(_ns), goto_functions(_goto_functions)
  {
  }

  void preprocess();
  void check();
  std::string translate();

protected:
  const namespacet &ns;
  goto_functionst goto_functions;
  std::list<symbolt> fun_decls;

  std::list<typet> global_types;
  std::list<symbolt> global_vars;
  std::map<std::string, exprt> global_const_initializers;
  std::map<std::string, exprt> global_static_initializers;

  std::map<std::string, std::list<typet>> local_types;
  std::map<std::string, std::list<symbolt>> local_static_vars;
  std::map<std::string, exprt> local_const_initializers;
  std::map<std::string, exprt> local_static_initializers;

  std::map<unsigned int, int> goto_scope_id;
  std::map<unsigned int, int> goto_parent_scope_id;

private:
  // Preprocessing methods
  void extract_symbol_tables();
  void extract_initializers();
  void assign_scope_ids(goto_programt &goto_program);
  void remove_unsupported_instructions(goto_programt &goto_program);
  exprt convert_array_assignment_to_function_call(code_assign2tc assign);
  typet get_base_type(typet type, namespacet ns);
  expr2tc get_base_expr(expr2tc expr);
  void sort_compound_types(const namespacet &ns, std::list<typet> &types);
  void sort_compound_types_rec(
    const namespacet &ns,
    std::list<typet> &sorted_types,
    std::set<typet> &observed_types,
    typet &type);
  void adjust_compound_assignments(goto_programt &goto_program);
  void adjust_compound_assignment_rec(
    goto_programt::instructionst &new_instructions,
    goto_programt::instructiont instruction,
    const namespacet &ns);

  // Translation methods
  std::string translate(goto_programt::instructiont instruction);
  std::string translate(goto_programt goto_program);
  std::string translate(goto_functiont goto_function);
};

#endif
