#include <goto2c/goto2c.h>
#include <clang-c-frontend/expr2ccode.h>
#include <clang-c-frontend/clang_c_language.h>
#include <util/expr_util.h>
#include <util/c_sizeof.h>

// Here we apply a sequence of preprocessing procedures
// to prepare the given GOTO program for translation
void goto2ct::preprocess()
{
  // Extracting all symbol tables
  extract_symbol_tables();
  // Extracting all initializers for variables with static lifetime
  extract_initializers();
  // Sorting all compound (i.e., struct/union) types (global and local)
  sort_compound_types(ns, global_types);
  for(auto &it : local_types)
    sort_compound_types(ns, local_types[it.first]);
  // Performing adjustments to each GOTO function individually
  for(auto &it : goto_functions.function_map)
  {
    remove_unsupported_instructions(it.second.body);
    // Dealing with assignments to struct, union, array variables
    adjust_compound_assignments(it.second.body);
    // Updating scope ids to the variables within the program
    it.second.body.compute_location_numbers();
    assign_scope_ids(it.second.body);
  }
}

// When dealing with "pointer" and "array" types
// and "typedef"s sometimes it is
// necessary to know the "underlying" type of the corresponding
// array/pointer type or typedef. This method returns such "underlying" type,
// or resolves an incomplete type if there is a corresponding type definition
// in the symbol table.
typet goto2ct::get_base_type(typet type, namespacet ns)
{
  if(type.id() == "array")
    return get_base_type(type.subtype(), ns);

  // This happens when the type is defined through a "typedef"
  // that cannot be reduced to one of the primitive types
  if(type.id() == "symbol")
  {
    const symbolt *symbol = ns.lookup(type.identifier());
    if(symbol && symbol->is_type && !symbol->type.is_nil())
      return get_base_type(symbol->type, ns);
  }

  // This is to deal with incomplete types and unions
  if(type.id() == "struct" || type.id() == "union")
  {
    const symbolt *symbol = ns.lookup(("tag-" + type.tag().as_string()));
    if(symbol && symbol->is_type && !symbol->type.is_nil())
      return symbol->type;
  }

  return type;
}

expr2tc goto2ct::get_base_expr(expr2tc expr)
{
  if(is_typecast2t(expr))
    return get_base_expr(to_typecast2t(expr).from);

  if(is_address_of2t(expr))
    return get_base_expr(to_address_of2t(expr).ptr_obj);

  if(is_index2t(expr))
    return get_base_expr(to_index2t(expr).source_value);

  return expr;
}

// This method recursivelly iterates through all members of the given
// struct/union "type" and builds a list "sorted_types" of all
// struct/union types appearing in "type" in the order where each
// successive element does not feature a struct/union member whose
// type has not already been included in the list.
void goto2ct::sort_compound_types_rec(
  const namespacet &ns,
  std::list<typet> &sorted_types,
  std::set<typet> &observed_types,
  typet &type)
{
  // Getting the base struct/union type in case we are dealing
  // with an array or a typedef
  typet base_type = get_base_type(type, ns);

  // The compound type have not been seen before by now
  if(observed_types.count(base_type) == 0)
  {
    observed_types.insert(base_type);
    // If "base_type" is struct/union, then iterate through its members.
    if(base_type.id() == "struct" || base_type.id() == "union")
    {
      struct_union_typet struct_union_type = to_struct_union_type(base_type);

      for(auto comp : struct_union_type.components())
        sort_compound_types_rec(ns, sorted_types, observed_types, comp.type());

      sorted_types.push_back(struct_union_type);
    }
  }
}

// This method iterates through the given list "types" of compound
// (i.e., struct/union) types and sorts them in the order they should
// be defined in the program.
void goto2ct::sort_compound_types(const namespacet &ns, std::list<typet> &types)
{
  std::list<typet> sorted_types;
  std::set<typet> observed_types;

  for(auto it = types.begin(); it != types.end(); it++)
    sort_compound_types_rec(ns, sorted_types, observed_types, *it);

  types = sorted_types;
}

void goto2ct::extract_symbol_tables()
{
  // Generating a list of input file names from the list
  // of input file paths
  std::unordered_set<std::string> input_files;
  for(auto it = clang_c_languaget::includes_map.begin();
      it != clang_c_languaget::includes_map.end();
      it++)
  {
    std::string filename = it->first.substr(it->first.find_last_of("/\\") + 1);
    input_files.insert(filename);
  }

  // Going throught the symbol table and initialising the program
  // data structures
  ns.get_context().foreach_operand_in_order(
    [this, &input_files](const symbolt &s) {
      // Skipping everything that appears in "esbmc_intrinsics.h"
      if(s.location.file().as_string() == "esbmc_intrinsics.h")
        return;

      // Extracting the name of the function where this symbol is declared.
      // It has a global scope if the location function name is empty.
      std::string sym_fun_name = s.location.function().as_string();
      // This is a type definition
      if(s.is_type)
      {
        if(sym_fun_name.empty())
          global_types.push_back(s.type);
        else
          local_types[sym_fun_name].push_back(s.type);
      }
      // This is a function declaration
      else if(s.type.id() == "code")
        fun_decls.push_back(s);
      // This is an extern or static variable
      else if(s.is_extern || s.static_lifetime)
      {
        // This is a global variabl
        if(sym_fun_name.empty())
          global_vars.push_back(s);
        else
          local_static_vars[sym_fun_name].push_back(s);
      }
    });
}

// We assume that all initializers are inside the "__ESBMC_main" function
void goto2ct::extract_initializers()
{
  // The program does not feature a goto function "__ESBMC_main"
  if(goto_functions.function_map.count("__ESBMC_main") == 0)
    return;

  if(goto_functions.function_map.count("c:@F@main") == 0)
    return;

  for(auto instr :
      goto_functions.function_map["__ESBMC_main"].body.instructions)
  {
    if(instr.type == ASSIGN)
    {
      code_assign2tc assign = to_code_assign2t(instr.code);
      if(is_symbol2t(assign->target))
      {
        const symbolt *sym = ns.lookup(to_symbol2t(assign->target).thename);
        exprt init = migrate_expr_back(get_base_expr(assign->source));
        if(sym && (sym->static_lifetime || sym->is_extern))
        {
          c_qualifierst c_qual(sym->type);
          std::string fun_name = sym->location.function().as_string();
          if(c_qual.is_constant)
            if(fun_name.empty())
              global_const_initializers[sym->id.as_string()] = init;
            else
              local_const_initializers[sym->id.as_string()] = init;
          else if(fun_name.empty())
          {
            global_static_initializers[sym->id.as_string()] = init;
            goto_functions.function_map["c:@F@main"]
              .body.instructions.push_front(instr);
          }
          else
            local_static_initializers[sym->id.as_string()] = init;
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
  // Fedor: first try to identify and separate DEAD clusters
  std::vector<std::string> scope_cluster;
  for(auto it = goto_program.instructions.rbegin();
      it != goto_program.instructions.rend();
      it++)
  {
    if(it->type == DEAD && std::next(it)->type == DEAD)
    {
      std::string sym_name = to_code_dead2t(it->code).value.as_string();
      std::string sym_name_short = expr2ccodet::get_name_shorthand(sym_name);
      if(
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
  for(auto it = goto_program.instructions.rbegin();
      it != goto_program.instructions.rend();
      it++)
  {
    unsigned int cur_scope_id = scope_ids_stack.back();
    unsigned int parent_scope_id = 0;
    if(scope_ids_stack.size() > 1)
      parent_scope_id = scope_ids_stack.at(scope_ids_stack.size() - 2);

    if(it->type == DEAD)
    {
      std::string sym_name = to_code_dead2t(it->code).value.as_string();
      // First we check if this symbol already appears in one of the scopes.
      bool in_a_scope = false;
      for(auto scope_syms : scope_syms_stack)
      {
        if(
          std::find(scope_syms.begin(), scope_syms.end(), sym_name) !=
          scope_syms.end())
        {
          in_a_scope = true;
          break;
        }
      }
      // If this symbol is not in the current scope already,
      // then it is added to the current stack.
      if(!in_a_scope)
      {
        // Adding this symbol to the current scope
        scope_syms_stack.back().push_back(sym_name);
        // If the previous instruction wasn't a DEAD
        // then we are entering a new scope.
        // Hence, creating a new unique scope ID
        if(
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
        if(std::next(it)->type != DEAD && scope_syms_stack.back().size() > 0)
        {
          scope_syms_stack.push_back({});
        }
      }
    }
    else if(it->type == DECL)
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
      if(cur_sym_it != scope_syms_stack.back().end())
        scope_syms_stack.back().erase(cur_sym_it);

      // Declarations of all declared variables within
      // this scope have been found. Hence, we are entering
      // the parent scope. So we can pop the current scope
      // from the stack. Unless it is the only scope on the stack
      // (i.e., all the declared symbols have function scope lifetime).
      if(scope_syms_stack.back().empty() && scope_syms_stack.size() > 1)
      {
        scope_syms_stack.pop_back();
        if(scope_ids_stack.size() > 1)
          scope_ids_stack.pop_back();
      }
      // Finally, push back empty set
      scope_syms_stack.push_back({});
    }
    it->scope_id = cur_scope_id;
    it->parent_scope_id = parent_scope_id;
    goto_scope_id[it->location_number] = cur_scope_id;
    goto_parent_scope_id[it->location_number] = parent_scope_id;
  }
}

void goto2ct::adjust_compound_assignment_rec(
  goto_programt::instructionst &new_instructions,
  goto_programt::instructiont instruction,
  const namespacet &ns)
{
  assert(is_code_assign2t(instruction.code));
  code_assign2tc assign = to_code_assign2t(instruction.code);
  // Assignment to a typecast
  if(is_typecast2t(assign->target))
  {
    expr2tc new_lhs = to_typecast2t(assign->target).from;
    code_assign2tc new_assign(new_lhs, assign->source);
    goto_programt::instructiont new_instruction(instruction);
    new_instruction.code = new_assign;

    adjust_compound_assignment_rec(new_instructions, new_instruction, ns);
  }
  // Assignment to a struct variable
  else if(
    is_symbol2t(assign->target) && is_struct_type(assign->target->type) &&
    is_constant_struct2t(assign->source))
  {
    typecast2tc type_cast(assign->target->type, assign->source);
    code_assign2tc new_assign(assign->target, type_cast);
    instruction.code = new_assign;
    new_instructions.push_back(instruction);
  }
  // Assignment to a union variable
  else if(
    is_symbol2t(assign->target) && is_union_type(assign->target->type) &&
    is_constant_union2t(assign->source))
  {
    typecast2tc type_cast(assign->target->type, assign->source);
    code_assign2tc new_assign(assign->target, type_cast);
    instruction.code = new_assign;
    new_instructions.push_back(instruction);
  }
  // Assignment to an array variable.
  // Turning it into a function call to "memcpy"
  /*
  else if(is_array_type(assign->target->type))
  {
    exprt fun_call = convert_array_assignment_to_function_call(assign);
    expr2tc fun_call2;
    migrate_expr(fun_call, fun_call2);
    instruction.code = fun_call2;
    instruction.type = FUNCTION_CALL;
    new_instructions.push_back(instruction);
  }
  */
  else
    new_instructions.push_back(instruction);
}

// Below we implement some temporary solutions for dealing with
// some issues introduced during GOTO conversion.
//
// 1) Assignments to typecasts. For example, GOTO often has
// constructs like (int) i = (int) i + 1, which come from the
// expressions like i += 1.
//
// 2) The variables initialisers are converted into assignments
// by the GOTO converter, and they become sintactically invalid
// assignments in C syntax. The below should be revisited when
// we decide on how to deal with declarations containing
// initialisers at the GOTO level.
void goto2ct::adjust_compound_assignments(goto_programt &goto_program)
{
  goto_programt::instructionst new_instructions;
  for(auto it = goto_program.instructions.begin();
      it != goto_program.instructions.end();
      it++)
  {
    if(it->type == ASSIGN)
    {
      adjust_compound_assignment_rec(new_instructions, *it, ns);
      // Now to preserve the tarets we cannot remove any instructions.
      // Hence, we replace the "value" in the original instruction
      // and insert the rest straight in.
      goto_programt::instructiont tmp_front = new_instructions.front();
      new_instructions.pop_front();
      it->code = tmp_front.code;
      goto_program.instructions.splice(it, new_instructions);
    }
  }
}

// This method iterates through the instructions in the given GOTO program
// and removes all GOTO instructions that cannot be currently
// translated
void goto2ct::remove_unsupported_instructions(goto_programt &goto_program)
{
  goto_programt::instructionst new_instructions;
  for(auto it = goto_program.instructions.begin();
      it != goto_program.instructions.end();
      it++)
  {
    // Unsupported assignment instructions
    if(it->type == ASSIGN)
    {
      code_assign2tc assign = to_code_assign2t(it->code);
      // Removing assignments to "dynamic_type2t"
      if(is_dynamic_size2t(assign->target))
        it = goto_program.instructions.erase(it);
    }
  }
}

exprt goto2ct::convert_array_assignment_to_function_call(code_assign2tc assign)
{
  // Creating a function call to the provided "memcpy" here
  // (with the empty left-hand side)
  code_function_callt function_call;
  // Creating a symbol for the "memcpy" function.
  // If the ESBMC version of "memcpy" is used in the program,
  // we just use it. Otherwise, we create a new symbol without adding
  // it to the symbol table.
  // (Probably the former is unnecessary! Need to test!)
  symbolt memcpy_sym;
  const symbolt *sym = ns.get_context().find_symbol("c:@F@memcpy");
  if(sym)
  {
    memcpy_sym = *sym;
  }
  else
  {
    memcpy_sym.id = "c:@F@memcpy";
    memcpy_sym.name = "c:@F@memcpy";
    memcpy_sym.type = function_call.type();
  }
  // Creating the arguments for the function call:
  //   1 - the array symbol being assigned to,
  //   2 - the typecasted compound argument (i.e., constant array)
  //     which value is being assigned,
  //   3 - the size of the constant array that is being assigned.
  exprt::operandst arguments;

  // Getting the base expression in case we assign
  // a typecast/address of/index of an expression
  expr2tc source = get_base_expr(assign->source);

  typecast2tc type_cast(assign->source->type, source);

  arguments.push_back(migrate_expr_back(assign->target));
  arguments.push_back(migrate_expr_back(type_cast));
  arguments.push_back(c_sizeof(migrate_type_back(assign->source->type), ns));
  // Populating the function call
  function_call.function() = symbol_expr(memcpy_sym);
  function_call.arguments() = arguments;
  function_call.location() = memcpy_sym.location;
  return function_call;
}
