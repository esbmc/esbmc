#include <goto2c/goto2c.h>
#include <clang-c-frontend/expr2ccode.h>
#include <clang-c-frontend/clang_c_language.h>
#include <util/expr_util.h>
#include <util/c_sizeof.h>

#include <iostream>

// Convert GOTO instruction to C
std::string goto2ct::translate(goto_programt::instructiont instruction)
{
  std::ostringstream out;
  
  // This is a GOTO label
  if(instruction.is_target())
    out << "__ESBMC_goto_label_" << instruction.target_number << ":; // Target\n";

  // Identifying the type of the GOTO instruction
  switch(instruction.type)
  {
  case ASSERT:
    out << expr2ccode(code_assertt(migrate_expr_back(instruction.guard)), ns) << " // "
        << instruction.location.comment() << "; ASSERT";
    break;
  case NO_INSTRUCTION_TYPE:
    out << "// NO_INSTRUCTION_TYPE";
    break;
  case GOTO:
    // If we know that the guard is TRUE, then just skip the "if" part
    if(!is_true(instruction.guard))
      out << "if(" << expr2ccode(migrate_expr_back(instruction.guard), ns) << ") ";

    out << "goto ";
    // Iterating through all label where this GOTO may lead
    for(goto_programt::instructiont::targetst::const_iterator gt_it = instruction.targets.begin();
        gt_it != instruction.targets.end();
        gt_it++)
    {
      if(gt_it != instruction.targets.begin())
        out << ", ";

      out << "__ESBMC_goto_label_" << (*gt_it)->target_number;
    }
    out << "; // GOTO";
    break;
  case FUNCTION_CALL:
    out << expr2ccode(migrate_expr_back(instruction.code), ns) << "; // FUNCTION_CALL";
    break;
  case RETURN:
    out << expr2ccode(migrate_expr_back(instruction.code), ns);
    break;
  case END_FUNCTION:
    out << "// END_FUNCTION: " << instruction.location.function();
    break;
  case DECL:
    assert(is_code_decl2t(instruction.code));
    // Not outputting void variables (e.g., something like "void i;").
    // Fedor: this should be removed through a program pass in the future.
    if(!is_empty_type(to_code_decl2t(instruction.code).type))
      out << expr2ccode(migrate_expr_back(instruction.code), ns) << "; // DECL";
    break;
  case DEAD:
    out << "// DEAD: " << expr2ccode(migrate_expr_back(instruction.code), ns);
    break;
  case OTHER:
    out << expr2ccode(migrate_expr_back(instruction.code), ns) << " // OTHER";
    break;
  case ASSIGN:
  {
    assert(is_code_assign2t(instruction.code));
    code_assign2tc assign = to_code_assign2t(instruction.code);
    if(is_array_type(assign->target->type))
    {
      exprt function_call = convert_array_assignment_to_function_call(assign);
      out << expr2ccode(function_call, ns) << "; // ARRAY INIT";
    }
    else if(
      is_dynamic_size2t(assign->target))
    {
      // Ignoring the dynamic size symbol for now
    }
    else
      out << expr2ccode(migrate_expr_back(assign), ns) << " // ASSIGN";

    break;
  }
  case ASSUME:
    out << "// ASSUME";
    break;
  case LOCATION:
    out << "// LOCATION " << instruction.location;
    break;
  case THROW:
    out << "// THROW";
    break;
  case CATCH:
    out << "// CATCH";
    break;
  case ATOMIC_BEGIN:
    out << "// ATOMIC_BEGIN";
    break;
  case ATOMIC_END:
    out << "// ATOMIC_END";
    break;
  case THROW_DECL:
    out << "// THROW_DECL";
    break;
  case THROW_DECL_END:
    out << "// THROW_DECL_END";
    break;
  case SKIP:
    out << "// SKIP";
    break;
  default:
    throw "unknown statement";
  }
  out << " // {" << instruction.scope_id << "}-{" << instruction.parent_scope_id << "}";
  if(instruction.parent_scope_id != 0)
    out << "-...";
  out << "\n";
  return out.str();
}

// Convert GOTO program to C
std::string goto2ct::translate(goto_programt goto_program)
{
  std::ostringstream out;
  
  std::vector<unsigned int> scope_ids_stack = {1};
  unsigned int cur_scope_id = scope_ids_stack.back();
  for(auto it = goto_program.instructions.begin(); it != goto_program.instructions.end(); it++)
  {
    //assert(it->scope_id == goto_scope_id[it->location_number]);
    //assert(it->parent_scope_id == goto_parent_scope_id[it->location_number]);
    // Scope change
    if(it->scope_id != cur_scope_id)
    {
      // Entering a new scope
      if(it->parent_scope_id == cur_scope_id)
      {
        out << "{ // SCOPE BEGIN {" << cur_scope_id << "}->{"
          << it->scope_id << "}\n";
        scope_ids_stack.push_back(it->scope_id);
        cur_scope_id = scope_ids_stack.back();
      }
      // Leaving the scope back to the current parent scope
      else
      {
        scope_ids_stack.pop_back();
        out << "} // SCOPE END {" << cur_scope_id << "}->{"
          << scope_ids_stack.back() << "}\n";
        cur_scope_id = scope_ids_stack.back();
        // If there two scopes next to each other (i.e., {inst1;}{inst2;})
        // we need to open a new scope immediately after the previous one
        // so that we do not skip through the first instruction
        // in the new scope
        if(it->scope_id != cur_scope_id)
        {
          out << "{ // SCOPE BEGIN {" << cur_scope_id << "}->{"
            << it->scope_id << "}\n";
          scope_ids_stack.push_back(it->scope_id);
          cur_scope_id = scope_ids_stack.back();
        }
      }
    }
    out << translate(*it);
  }
  return out.str();
}

// This translates the given GOTO program (aka list of GOTO functions)
std::string goto2ct::translate()
{
  std::ostringstream out;
  
  // Starting the translation
  out << "///////////////////////////////////////////////////////\n";
  out << "//\n";
  out << "// This program is generated by the GOTO2C translator\n";
  out << "//\n";
  out << "///////////////////////////////////////////////////////\n";
  out << "\n\n\n";

  // Some additional includes
  out << "///////////////////   HEADERS   ///////////////////////\n";

  // To check if the "assert" function is ever declared
  const symbolt *assert_sym = ns.lookup("c:@F@assert");
  // To check if the "assert" function is ever defined
  auto assert_it = goto_functions.function_map.find("c:@F@assert");

  // This is a temporary hack. We need a more elegant way of solving this.
  if(!assert_sym || (assert_sym && !assert_it->second.body_available))
  {
    out << "#include <assert.h> // included automatically\n";
  }

  // Includes appearing in the source files
  for(auto it = clang_c_languaget::includes_map.begin();
      it != clang_c_languaget::includes_map.end();
      it++)
  {
    for(auto include : it->second)
    {
      if(!include.empty())
      {
        out << "#include <" << include << "> // included in \"" << it->first
            << "\"\n";
      }
    }
    out << "\n";
  }
  out << "\n\n";

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

  // Types
  out << "/////////////////////   GLOBAL TYPES  ////////////////////////\n";
  for(auto type : global_types)
  {
    if(type.id() == "struct" || type.id() == "union")
      out << typedef2ccode(to_struct_union_type(type), ns) << ";\n";
    else
      out << type2ccode(type, ns) << ";\n";
  }
  out << "\n\n";

  // Global variables
  out << "/////////////////////   GLOBAL VARIABLES  ////////////////////////\n";
  for(auto s : global_vars)
  {
    // Looking first if there is an initializer
    if(global_const_initializers.count(s.id.as_string()) > 0)
      out << expr2ccode(code_declt(symbol_expr(s), global_const_initializers[s.id.as_string()]), ns) << "; // " << s.location << "\n";
    else
      out << expr2ccode(code_declt(symbol_expr(s)), ns) << "; // " << s.location << "\n";
  }
  out << "\n\n";

  // Declarations
  out << "/////////////////  FUNCTION DECLARATIONS   ////////////////////\n";
  for(auto s : fun_decls)
    out << expr2ccode(code_declt(symbol_expr(s)), ns) << "; // " << s.location << "\n";
  out << "\n\n";

  // Definitions
  out << "/////////////////   FUNCTION DEFINITIONS   ////////////////////\n";
  // Iterating through the available GOTO functions
  for(auto &it : goto_functions.function_map)
  {
    // First making sure that the GOTO function name can
    // be found in the symbol table
    const symbolt *symbol = ns.lookup(it.first);
    assert(symbol);
    // Ignoring all the functions that are not defined 
    // in the source files
    if(
      input_files.find(symbol->location.file().as_string()) ==
      input_files.end())
      continue;
    // Checking if the function has a definition or not
    if(it.second.body_available)
    {
      std::string fun_name_short = expr2ccodet::get_name_shorthand(it.first.as_string());
      // Using the location of the first instruction in the function body
      locationt loc = it.second.body.instructions.front().location;
      out << "// From \"" << loc << "\"\n";
      // Converting the function declaration part first
      code_declt decl(symbol_expr(*symbol));
      out << expr2ccode(decl, ns) << "\n";
      // Converting the function body
      out << "{\n";
      out << "/////////////////////   LOCAL TYPES  ////////////////////////\n";
      for(auto type : local_types[expr2ccodet::get_name_shorthand(it.first.as_string())])
      {
        if(type.id() == "struct" || type.id() == "union")
          out << typedef2ccode(to_struct_union_type(type), ns) << ";\n";
        else
          out << type2ccode(type, ns) << ";\n";
      }
      out << "///////////////   LOCAL STATIC VARIABLES  //////////////////\n";
      for(auto s : local_static_vars[fun_name_short])
      {
        if(local_static_initializers.count(s.id.as_string()) > 0)
          out << expr2ccode(code_declt(symbol_expr(s), local_static_initializers[s.id.as_string()]), ns) << "; // " << s.location << "\n";
        else
          out << expr2ccode(code_declt(symbol_expr(s)), ns) << "; // " << s.location << "\n";
      }
      // Converting the rest of the body
      out << translate(it.second.body);
      out << "}\n\n";
    }
  }
  return out.str();
}
