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

  // Preprocessing methods
  void preprocess();
  void preprocess(goto_functionst &goto_functions);
  void preprocess(std::string function_id, goto_functiont &goto_function);
  void preprocess(goto_programt &goto_program);
  void preprocess(goto_programt::instructiont &instruction);

  // Checking methods
  void check();
  void check(goto_functionst &goto_functions);
  void check(std::string function_id, goto_functiont &goto_function);
  void check(goto_programt &goto_program);
  void check(goto_programt::instructiont &instruction);

  // Translation methods
  std::string translate();
  std::string translate(goto_functionst &goto_functions);
  std::string translate(std::string function_id, goto_functiont &goto_function);
  std::string translate(goto_programt &goto_program);
  std::string translate(goto_programt::instructiont &instruction);

  // Access methods
  goto_functionst get_goto_functions()
  {
    return goto_functions;
  }

  namespacet get_namespace()
  {
    return ns;
  }

protected:
  namespacet &ns;
  goto_functionst goto_functions;

  std::list<symbolt> fun_decls;
  std::list<typet> global_types;
  std::list<symbolt> global_vars;
  std::list<symbolt> extern_vars;
  std::map<std::string, exprt> initializers;

  std::map<std::string, std::list<typet>> local_types;
  std::map<std::string, std::list<symbolt>> local_static_vars;

private:
  // Auxiliary methods
  typet get_base_type(typet type, namespacet ns);

  // Preprocessing methods
  void extract_symbol_tables();
  void extract_initializers_from_esbmc_main();
  void simplify_initializers();
  void sort_compound_types(namespacet &ns, std::list<typet> &types);
  void sort_compound_types_rec(
    namespacet &ns,
    std::list<typet> &sorted_types,
    std::set<typet> &observed_types,
    typet &type);
  void assign_scope_ids(goto_programt &goto_program);
  void remove_unsupported_instructions(goto_programt &goto_program);
  expr2tc replace_array_assignment_with_memcpy(const code_assign2t &assign);
  void adjust_invalid_assignments(goto_programt &goto_program);
  void adjust_invalid_assignment_rec(
    goto_programt::instructionst &new_instructions,
    goto_programt::instructiont instruction,
    namespacet &ns);

  // Checking methods for each individual instruction type
  void check_assert(goto_programt::instructiont instruction);
  void check_assume(goto_programt::instructiont instruction);
  void check_goto(goto_programt::instructiont instruction);
  void check_function_call(goto_programt::instructiont instruction);
  void check_return(goto_programt::instructiont instruction);
  void check_end_function(goto_programt::instructiont instruction);
  void check_decl(goto_programt::instructiont instruction);
  void check_dead(goto_programt::instructiont instruction);
  void check_assign(goto_programt::instructiont instruction);
  void check_location(goto_programt::instructiont instruction);
  void check_skip(goto_programt::instructiont instruction);
  void check_throw(goto_programt::instructiont instruction);
  void check_catch(goto_programt::instructiont instruction);
  void check_atomic_begin(goto_programt::instructiont instruction);
  void check_atomic_end(goto_programt::instructiont instruction);
  void check_throw_decl(goto_programt::instructiont instruction);
  void check_throw_decl_end(goto_programt::instructiont instruction);
  void check_other(goto_programt::instructiont instruction);

  // Methods for checking expressions
  void check_guard(expr2tc guard);
};

#endif
