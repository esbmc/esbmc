#include <goto2c/goto2c.h>
#include <goto2c/expr2c.h>
#include <util/expr_util.h>
#include <util/c_sizeof.h>

// Here we apply a sequence of preprocessing procedures
// to prepare the given GOTO program for translation
void goto2ct::preprocess()
{
  // Extracting all symbol tables
  extract_symbol_tables();
  // Extracting all initializers for variables with static lifetime
  extract_initializers_from_esbmc_main();
  // Simplifying initializers
  simplify_initializers();
  // Sorting all compound (i.e., struct/union) types (global and local)
  sort_compound_types(ns, global_types);
  for (auto &it : local_types)
    sort_compound_types(ns, local_types[it.first]);
  // Preprocessing the functions now
  preprocess(goto_functions);
}

void goto2ct::preprocess(goto_functionst &goto_functions)
{
  // Remove ESBMC main first
  goto_functions.function_map.erase("__ESBMC_main");
  // Iterating through all GOTO functions
  for (auto it = goto_functions.function_map.begin();
       it != goto_functions.function_map.end();)
  {
    // Preprocess the function if its body is available
    if (it->second.body_available)
    {
      preprocess(it->first.as_string(), it->second);
      ++it;
    }
    // Remove the function otherwise
    else
      goto_functions.function_map.erase(it++);
  }
}

void goto2ct::preprocess(
  std::string function_id [[maybe_unused]],
  goto_functiont &goto_function)
{
  preprocess(goto_function.body);
}

void goto2ct::preprocess(goto_programt &goto_program)
{
  // Removing instructions that cannot be currently translated
  remove_unsupported_instructions(goto_program);
  // Dealing with assignments to struct, union, array variables
  adjust_invalid_assignments(goto_program);
  // Updating scope ids to the variables within the program
  assign_scope_ids(goto_program);
}

// When dealing with "pointer" and "array" types
// and "typedef"s sometimes it is
// necessary to know the "underlying" type of the corresponding
// array/pointer type or typedef. This method returns such "underlying" type,
// or resolves an incomplete type if there is a corresponding type definition
// in the symbol table.
typet goto2ct::get_base_type(typet type, namespacet ns)
{
  if (type.id() == "array")
    return get_base_type(type.subtype(), ns);

  // This happens when the type is defined through a "typedef"
  // that cannot be reduced to one of the primitive types
  if (type.id() == "symbol")
  {
    const symbolt *symbol = ns.lookup(type.identifier());
    if (symbol && symbol->is_type && !symbol->type.is_nil())
      return get_base_type(symbol->type, ns);
  }

  // This is to deal with incomplete types and unions
  if (type.id() == "struct" || type.id() == "union")
  {
    const symbolt *symbol = ns.lookup(("tag-" + type.tag().as_string()));
    if (symbol && symbol->is_type && !symbol->type.is_nil())
      return symbol->type;
  }

  return type;
}

// This method recursively iterates through all members of the given
// struct/union "type" and builds a list "sorted_types" of all
// struct/union types appearing in "type" in the order where each
// successive element does not feature a struct/union member whose
// type has not already been included in the list.
void goto2ct::sort_compound_types_rec(
  namespacet &ns,
  std::list<typet> &sorted_types,
  std::set<typet> &observed_types,
  typet &type)
{
  // Getting the base struct/union type in case we are dealing
  // with an array or a typedef
  typet base_type = get_base_type(type, ns);

  // The compound type have not been seen before by now
  if (observed_types.count(base_type) == 0)
  {
    observed_types.insert(base_type);
    // If "base_type" is struct/union, then iterate through its members.
    if (base_type.id() == "struct" || base_type.id() == "union")
    {
      struct_union_typet struct_union_type = to_struct_union_type(base_type);

      for (auto comp : struct_union_type.components())
        sort_compound_types_rec(ns, sorted_types, observed_types, comp.type());

      sorted_types.push_back(struct_union_type);
    }
  }
}

// This method iterates through the given list "types" of compound
// (i.e., struct/union) types and sorts them in the order they should
// be defined in the program.
void goto2ct::sort_compound_types(namespacet &ns, std::list<typet> &types)
{
  std::list<typet> sorted_types;
  std::set<typet> observed_types;

  for (auto it = types.begin(); it != types.end(); it++)
    sort_compound_types_rec(ns, sorted_types, observed_types, *it);

  types = sorted_types;
}

// This method simply goes through all symbols in the program and places
// them into separate lists/tables for easier/quicker access.
void goto2ct::extract_symbol_tables()
{
  // Going through the symbol table
  ns.get_context().foreach_operand_in_order([this](const symbolt &s) {
    // Skipping everything that appears in "esbmc_intrinsics.h"
    // or with an empty location.
    if (
      s.location.file().as_string() == "esbmc_intrinsics.h" ||
      s.location.as_string() == "")
      return;

    // Extracting the name of the function where this symbol is declared.
    // The symbol will be considered to belong to global scope if
    // the location function name is empty.
    std::string sym_fun_name = s.location.function().as_string();

    // This is a type definition
    if (s.is_type)
    {
      if (sym_fun_name.empty())
        global_types.push_back(s.type);
      else
        local_types[sym_fun_name].push_back(s.type);
    }
    // This is a function declaration
    else if (s.type.id() == "code")
      fun_decls.push_back(s);
    // This is an extern variable
    else if (s.is_extern)
      extern_vars.push_back(s);
    // This is a static variable
    else if (s.static_lifetime)
    {
      // This is a global variable
      if (sym_fun_name.empty())
        global_vars.push_back(s);
      else
        local_static_vars[sym_fun_name].push_back(s);
    }
  });
}

// This methods tries to resolve one step in the chains of initializers.
//
// In other words, the following "two-step" initialization
// (which is also illegal in C, but allowed in GOTO programs
// that ESBMC produces):
//
//  const static int tmp = 10;
//  const static int b = tmp;
//
// is converted to:
//
//  const static int a = 10;
//  const static int b = 10;
//
// This will not resolve chains with 3 and more initializers,
// which never seem to appear in our GOTO programs.
void goto2ct::simplify_initializers()
{
  for (auto init : initializers)
  {
    if (init.second.id() == "symbol")
    {
      std::string init_sym = init.second.identifier().as_string();
      if (initializers.count(init_sym) > 0)
        initializers[init.first] = initializers[init_sym];
    }
  }
}

// This method is to deal with ESBMC splitting variables declarations
// with initializers into two separate instructions: 1) declaration
// without an initializer, 2) assignment of the initializer to the declared
// variable. Some of the issues caused by this include:
// initialization of arrays, and constant variables.
// Here we extract all initializers that appear
// inside the "__ESBMC_main" function. Note that they
// include initializers for global and local static variables.
// This method however, does not deal with locating initializers
// for local variables.
void goto2ct::extract_initializers_from_esbmc_main()
{
  // The program does not feature a goto function "__ESBMC_main"
  if (goto_functions.function_map.count("__ESBMC_main") == 0)
    return;

  for (auto instr :
       goto_functions.function_map["__ESBMC_main"].body.instructions)
  {
    if (instr.type == ASSIGN)
    {
      const code_assign2t &assign = to_code_assign2t(instr.code);
      if (is_symbol2t(assign.target))
      {
        const symbolt *sym = ns.lookup(to_symbol2t(assign.target).thename);
        exprt init = migrate_expr_back(assign.source);
        if (sym && sym->static_lifetime)
        {
          initializers[sym->id.as_string()] = init;
        }
      }
    }
  }
}

// This function assigns a unique scope ID to each GOTO instruction
// within the program. Such ID's are useful for identifying
// the lifetime of each declared variable within the program
// which is necessary to produce a correct translation into
// languages like C/C++.
// GOTO syntax does provide the means of solving the above problem
// by tracking all DECL instructions and the corresponding DEAD
// instructions. Here we use this information to preform some additional
// analysis for cases when there are multiple DEAD instructions
// for a single DECL (e.g., when there "break"'s inside the loop, GOTO
// converter will declare all loop local variables as DEAD at every
// exit point of the loop). Moreover, there may be clusters of
// DECL and DEAD instructions which means that we can
// place all corresponding variables in the same scope, instead
// of declaring a new scope for each individual pairs of DECL and DEAD.
// Finally, the produced scope ID's may be useful for various
// analyses prior to and during symbolic execution.
void goto2ct::assign_scope_ids(goto_programt &goto_program)
{
  // First try to identify and separate DEAD clusters
  std::vector<std::string> scope_cluster;
  for (auto it = goto_program.instructions.rbegin();
       it != goto_program.instructions.rend();
       it++)
  {
    if (it->type == DEAD && std::next(it)->type == DEAD)
    {
      std::string sym_name = to_code_dead2t(it->code).value.as_string();
      std::string sym_name_short = expr2ct::get_name_shorthand(sym_name);
      if (
        std::find(scope_cluster.begin(), scope_cluster.end(), sym_name_short) !=
        scope_cluster.end())
      {
        scope_cluster.clear();
        goto_programt::instructiont skip;
        skip.type = SKIP;
        skip.location = it->location;
        goto_program.instructions.insert(it.base(), skip);
      }
      else
        scope_cluster.push_back(sym_name_short);
    }
  }

  // All variables with the function lifetime
  // get a scope id of 1 by default, while
  // the default scope ID value for each instruction is 0,
  // which defines the global scope.

  // This variable is used for generating unique scope ID's
  unsigned int scope_count = 1;
  // Stack containing the sequence of all parent scope ID's
  // relative to the current scope. As per above, the parent
  // scope ID for all GOTO instruction within a body of
  // a GOTO program is equal to 1.
  std::vector<unsigned int> scope_ids_stack = {scope_count};
  // List of all scopes recorded until the current instruction is processed.
  std::vector<std::vector<std::string>> scope_syms_stack = {{}, {}};
  // Iterating through the GOTO instructions in reverse order
  for (auto it = goto_program.instructions.rbegin();
       it != goto_program.instructions.rend();
       it++)
  {
    unsigned int cur_scope_id = scope_ids_stack.back();
    unsigned int parent_scope_id = 0;
    if (scope_ids_stack.size() > 1)
      parent_scope_id = scope_ids_stack.at(scope_ids_stack.size() - 2);

    if (it->type == DEAD)
    {
      std::string sym_name = to_code_dead2t(it->code).value.as_string();
      // First we check if this symbol already appears in one of the scopes.
      bool in_a_scope = false;
      for (auto scope_syms : scope_syms_stack)
      {
        if (
          std::find(scope_syms.begin(), scope_syms.end(), sym_name) !=
          scope_syms.end())
        {
          in_a_scope = true;
          break;
        }
      }
      // If this symbol is not in the current scope already,
      // then it is added to the current stack.
      if (!in_a_scope)
      {
        // Adding this symbol to the current scope
        scope_syms_stack.back().push_back(sym_name);
        // If the previous instruction wasn't a DEAD
        // then we are entering a new scope.
        // Hence, creating a new unique scope ID
        if (
          std::prev(it)->type != DEAD &&
          it != goto_program.instructions.rbegin())
        {
          parent_scope_id = scope_ids_stack.at(scope_ids_stack.size() - 1);
          scope_ids_stack.push_back(++scope_count);
          cur_scope_id = scope_ids_stack.back();
        }
        // If the next instruction is outside the current
        // cluster of DEAD instructions,
        // then we push the current list of scope symbols
        // to the list of scopes and clear the current scope.
        // Otherwise, just continue.
        // !!! Also probably need to check whether it == instructions.rend()
        if (std::next(it)->type != DEAD && scope_syms_stack.back().size() > 0)
        {
          scope_syms_stack.push_back({});
        }
      }
    }
    else if (it->type == DECL)
    {
      // Firstly, pop_back from the scope stack
      scope_syms_stack.pop_back();
      // Getting the ID for the current scope
      cur_scope_id = scope_ids_stack.back();
      // Looking up if the declared symbol is within the current scope
      std::string sym_name = to_code_decl2t(it->code).value.as_string();
      auto cur_sym_it = std::find(
        scope_syms_stack.back().begin(),
        scope_syms_stack.back().end(),
        sym_name);

      // Removing the symbol from the current list of DEAD
      // variables since the first corresponding DECL
      // for this symbol has been found.
      if (cur_sym_it != scope_syms_stack.back().end())
        scope_syms_stack.back().erase(cur_sym_it);

      // Declarations of all declared variables within
      // this scope have been found. Hence, we are entering
      // the parent scope. So we can pop the current scope
      // from the stack. Unless it is the only scope on the stack
      // (i.e., all the declared symbols have function scope lifetime).
      if (scope_syms_stack.back().empty() && scope_syms_stack.size() > 1)
      {
        scope_syms_stack.pop_back();
        if (scope_ids_stack.size() > 1)
          scope_ids_stack.pop_back();
      }
      // Finally, push back empty set
      scope_syms_stack.push_back({});
    }
    it->scope_id = cur_scope_id;
    it->parent_scope_id = parent_scope_id;
  }
}

// Recursive method for removing invalid assignments
//
// 1) Assignments to typecasts. For example, GOTO often has
// constructs like (int) i = (int) i + 1, which come from adjusting
// expressions like i += 1.
//
// 2) The variables initializers are converted into assignments
// by the GOTO converter, and they become syntactically invalid
// assignments in C syntax.
void goto2ct::adjust_invalid_assignment_rec(
  goto_programt::instructionst &new_instructions,
  goto_programt::instructiont instruction,
  namespacet &ns)
{
  assert(is_code_assign2t(instruction.code));
  const code_assign2t &assign = to_code_assign2t(instruction.code);
  // Assignment to a typecast
  if (is_typecast2t(assign.target))
  {
    expr2tc new_lhs = to_typecast2t(assign.target).from;
    expr2tc new_assign = code_assign2tc(new_lhs, assign.source);
    goto_programt::instructiont new_instruction(instruction);
    new_instruction.code = new_assign;

    adjust_invalid_assignment_rec(new_instructions, new_instruction, ns);
  }
  // Assignment to an array variable.
  // Turning it into a function call to "memcpy"
  else if (is_array_type(assign.target->type))
  {
    expr2tc fun_call = replace_array_assignment_with_memcpy(assign);
    instruction.code = fun_call;
    instruction.type = FUNCTION_CALL;
    new_instructions.push_back(instruction);
  }
  else
    new_instructions.push_back(instruction);
}

// This method adjusts assignments that appear in GOTO, but
// their direct translation into C/C++ syntax produces
// invalid assignments.
//
// Note, that each such "invalid assignment" instruction may be
// transformed into multiple instructions.
// See "goto2ct::adjust_invalid_assignment_rec" for more information.
void goto2ct::adjust_invalid_assignments(goto_programt &goto_program)
{
  goto_programt::instructionst new_instructions;
  for (auto it = goto_program.instructions.begin();
       it != goto_program.instructions.end();
       it++)
  {
    if (it->type == ASSIGN)
    {
      // Apply the recursive method first to the given ASSIGN instruction.
      adjust_invalid_assignment_rec(new_instructions, *it, ns);
      // Now to preserve the targets we cannot remove any instructions.
      // Hence, we replace the "value" in the original instruction
      // and insert the rest straight in.
      goto_programt::instructiont tmp_front = new_instructions.front();
      new_instructions.pop_front();
      it->code = tmp_front.code;
      it->type = tmp_front.type;
      goto_program.instructions.splice(it, new_instructions);
    }
  }
}

// This method iterates through the instructions in the given GOTO program
// and removes all GOTO instructions that cannot be currently
// translated into C/C++.
void goto2ct::remove_unsupported_instructions(goto_programt &goto_program)
{
  goto_programt::instructionst new_instructions;
  for (auto it = goto_program.instructions.begin();
       it != goto_program.instructions.end();
       it++)
  {
    if (it->type == ASSIGN)
    {
      const code_assign2t &assign = to_code_assign2t(it->code);
      // Removing assignments to "dynamic_type2t"
      if (is_dynamic_size2t(assign.target))
        it = goto_program.instructions.erase(it);
    }
  }
}

// This methods replaces assignments to array symbols
// (which are not allowed in C) with function calls to "memcpy".
// For example,
//
//      "array1 = array2;"
//
// is replaced by
//
//      "memcpy(array1, array2, <size_of_array2>);"
//
expr2tc
goto2ct::replace_array_assignment_with_memcpy(const code_assign2t &assign)
{
  // Creating a compound literal from the RHS
  expr2tc type_cast = typecast2tc(assign.source->type, assign.source);

  // Creating a "void*" type frequently used below
  type2tc pointertype = pointer_type2tc(get_empty_type());

  // Creating a function signature here: arguments types,
  // return type, function arguments names, function name
  std::vector<type2tc> args = {pointertype, pointertype, get_uint64_type()};
  type2tc ret_type = get_empty_type();
  std::vector<irep_idt> arg_names = {"d", "s", "n"};
  type2tc fun_type = code_type2tc(args, pointertype, arg_names, false);
  expr2tc fun_sym = symbol2tc(fun_type, "c:@F@memcpy");

  // Now defining the function inputs
  std::vector<expr2tc> ops = {
    assign.target, type_cast, c_sizeof(assign.source->type, ns)};

  // Returning a function call (with the empty left-hand side)
  // to the newly generated "memcpy" function with the inputs
  // defined in "ops".
  expr2tc fun_call = code_function_call2tc(expr2tc(), fun_sym, ops);

  return fun_call;
}
