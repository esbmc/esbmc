#include <goto-symex/execution_trace.h>
#include <goto-symex/goto_symex.h>
#include <clang-c-frontend/expr2ccode.h>
#include <goto-programs/loopst.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/base_type.h>
#include <pointer-analysis/dereference.h>

#include <iostream>
#include <regex>

std::string get_function_name(expr2tc func_call)
{
  assert(is_code_function_call2t(func_call));
  code_function_call2tc code_function_call = to_code_function_call2t(func_call);
  symbol2t func_call_sym = to_symbol2t(code_function_call->function);
  std::string fun_name = func_call_sym.thename.c_str();
  size_t at_pos = fun_name.find_last_of('@');
  if(at_pos != std::string::npos)
    return fun_name.substr(at_pos + 1, fun_name.size() - at_pos);

  return fun_name;
}

expr2tc get_symbol(const expr2tc &expr)
{
  if(is_member2t(expr))
    return get_symbol(to_member2t(expr).source_value);
  if(is_index2t(expr))
    return get_symbol(to_index2t(expr).source_value);

  return expr;
}

type2tc get_base_decl_type(type2tc type)
{
  if(is_pointer_type(type))
  {
    pointer_type2tc ptr_type = to_pointer_type(type);
    return get_base_decl_type(ptr_type->subtype);
  }
  if(is_array_type(type))
  {
    array_type2tc arr_type = to_array_type(type);
    return get_base_decl_type(arr_type->subtype);
  }

  return type;
}

expr2tc get_base_expr(expr2tc expr)
{
  if(is_typecast2t(expr))
    return get_base_expr(to_typecast2t(expr).from);

  return expr;
}


std::string c_instructiont::convert_to_c(namespacet &ns)
{
  std::ostringstream out;

  unsigned int loop_unwind = atoi(config.options.get_option("unwind").c_str());

  //out << "// location = " << location << "; scope_id = " << scope_id << "\n";

  if(is_target())
  {
    std::string func_name = location.get_function().c_str();
    out << "__ESBMC_goto_label_" << func_name << "_" << fun_call_nums[func_name]
        << "_" << target_number;

    out << ":; // Target\n";
  }

  switch(type)
  {
  case ASSERT:
    out << convert_assert_to_c(ns) << " // ASSERT";
    break;
  case NO_INSTRUCTION_TYPE:
    out << "// NO_INSTRUCTION_TYPE";
    break;
  case GOTO:
    if(!is_true(guard))
      out << "if(" << expr2ccode(migrate_expr_back(guard), ns) << ") ";

    out << "goto ";

    for(instructiont::targetst::const_iterator gt_it = targets.begin();
        gt_it != targets.end();
        gt_it++)
    {
      if(gt_it != targets.begin())
        out << ", ";

      std::string func_name = location.get_function().c_str();
      out << "__ESBMC_goto_label_" << func_name << "_"
          << fun_call_nums[func_name] << "_" << (*gt_it)->target_number;
    }
    out << "; // GOTO";
    break;
  case FUNCTION_CALL:
  {
    std::string func_name = get_function_name(code);
    fun_call_nums[func_name]++;
    out << "// FUNCTION_CALL " << expr2ccode(migrate_expr_back(code), ns);
  }
  break;
  case RETURN:
  {
    std::string func_name = location.get_function().c_str();
    if(is_code_assign2t(code))
      out << expr2ccode(migrate_expr_back(code), ns)
          << " // assign RETURN value\n";
    out << "goto __ESBMC_end_of_function_label_" << func_name << "_"
        << fun_call_nums[func_name] << ";";
    out << " // RETURN from " << func_name;
  }
  break;
  case END_FUNCTION:
  {
    std::string func_name = location.get_function().c_str();
    out << "__ESBMC_end_of_function_label_" << func_name << "_"
        << fun_call_nums[func_name] << ":;\n";
    out << "// END_FUNCTION " << func_name;
  }
  break;
  case DECL:
    out << convert_decl_to_c(ns) << " // DECL";
    break;
  case DEAD:
    out << "// DEAD: " << expr2ccode(migrate_expr_back(code), ns);
    break;
  case OTHER:
    out << convert_other_to_c(ns) << " // OTHER";
    break;
  case ASSIGN:
  {
    assert(is_code_assign2t(code));
    code_assign2tc assign = to_code_assign2t(code);
    out << expr2ccode(migrate_expr_back(code), ns) << " // ASSIGN";
  }
  break;
  case ASSUME:
    out << "// ASSUME";
    break;
  case LOCATION:
    out << "// LOCATION " << location;
    out << " /*" << expr2ccode(migrate_expr_back(code), ns) << "*/";
    out << " /*" << expr2ccode(migrate_expr_back(guard), ns) << "*/";
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
    if(scope_begin)
      out << "{ // SCOPE_BEGIN";
    else if(scope_end)
      out << "} // SCOPE_END";
    else
      out << "// SKIP";
    break;
  default:
    throw "unknown statement";
  }

  return out.str();
}

void replace_esbmc_alloc_size(expr2tc &expr)
{
  //expr->Foreach_operand([](expr2tc e)
  for(unsigned int i = 0; i < expr->get_num_sub_exprs(); i++)
  {
    expr2tc *e = expr->get_sub_expr_nc(i);
    if(is_index2t(*e))
    {
      index2tc idx = to_index2t(*e);
      if(to_symbol2t(idx->source_value).thename.as_string() == "c:@__ESBMC_alloc_size")
      {
        //std::cerr << ">>>>> index = " << idx->index << "\n";
        if(is_pointer_object2t(idx->index))
        {
          pointer_object2tc ptr_obj = to_pointer_object2t(idx->index);
          //std::cerr << ">>>>> ptr_object = " << ptr_obj << "\n";
          if(is_address_of2t(*(ptr_obj->get_sub_expr(0))))
          {
            address_of2tc addr = to_address_of2t(*(ptr_obj->get_sub_expr(0)));
            //std::cerr << ">>>>> addr = " << addr->ptr_obj << "\n";
            if(is_array_type(addr->ptr_obj->type))
            {
              array_type2tc arr_type = to_array_type(addr->ptr_obj->type); 
              unsigned int arr_subtype_size = type_byte_size(arr_type->subtype).to_uint64();
              // this arr_size is equal to the number of elements of the given array
              // array size in bytes
              expr2tc arr_size = mul2tc(arr_type->array_size->type, arr_type->array_size, gen_ulong(arr_subtype_size));
              std::cerr << ">>>>> arr_size = " << arr_size << "\n";
              /*
              greaterthan2tc gt(addr->ptr_obj, same->side_1);
              greaterthan2tc gt2(add2tc(same->side_1->type, same->side_1, gen_ulong(1)), 
                                  add2tc(arr_type->subtype, addr->ptr_obj, arr_size));
              or2tc in_bounds(gt, gt2);
              simplify(in_bounds);
              return convert(migrate_expr_back(in_bounds), precedence);
              */
              e->swap(arr_size);
              std::cerr << ">>>>> e after swap = " << *e << "\n";
              std::cerr << ">>>>> expr = " << expr << "\n";
              std::cerr << "----------\n";
            }
          }
        }
      }
    }
    else
      replace_esbmc_alloc_size(*e);
  //});
  }
}

std::string c_instructiont::convert_assert_to_c(namespacet &ns)
{
  replace_esbmc_alloc_size(guard);
  std::ostringstream out;
  out << "assert((" << expr2ccode(migrate_expr_back(guard), ns)
      << ") && \"[what: " << msg << "] [location: " << location << "]\");";

  //replace_esbmc_alloc_size(guard);

  return out.str();
}

std::string c_instructiont::convert_decl_to_c(namespacet &ns)
{
  assert(is_code_decl2t(code));
  std::ostringstream out;
  // Before converting the variable declaration we need to check
  // if the declared variable contains an initialiser. Such declaration
  // will be stored in the "decl" variable of type "exprt" instead
  // of the regular "expr2t" since a "code_decl2tc" loses the
  // initialiser when migrated to "code_declt".
  if(decl.operands().size() > 0)
    out << expr2ccode(decl, ns);
  else
    out << expr2ccode(migrate_expr_back(code), ns);

  return out.str();
}

std::string c_instructiont::convert_other_to_c(namespacet &ns)
{
  std::ostringstream out;
  if(is_code_printf2t(code))
  {
    out << expr2ccode(migrate_expr_back(code), ns);
    return out.str();
  }
  else if(is_code_free2t(code))
  {
    out << expr2ccode(migrate_expr_back(code), ns);
    return out.str();
  }
  else
  {
    out << "/* " << expr2ccode(migrate_expr_back(code), ns) << " */\n";
    //out << "/* " << code << " */\n";
  }
  return out.str();
}

unsigned int c_instructiont::get_loop_depth()
{
  return loop_depth;
}

void c_instructiont::set_loop_depth(unsigned int depth)
{
  loop_depth = depth;
}

void c_instructiont::increment_loop_depth()
{
  loop_depth++;
}

std::vector<c_instructiont>
inline_function_call(c_instructiont func_call, goto_functiont function)
{
  std::vector<c_instructiont> res;
  assert(is_code_function_call2t(func_call.code));
  code_function_call2tc code_function_call = to_code_function_call2t(func_call.code);
  assert(
    function.type.arguments().size() ==
    code_function_call->operands.size());
  // Going through all the function arguments
  for(unsigned int i = 0; i < function.type.arguments().size(); i++)
  {
    // Declaration
    code_typet::argumentt arg = function.type.arguments()[i];
    const irep_idt &identifier = arg.get_identifier();
    symbol2tc lhs(migrate_type(arg.type()), identifier);
    code_decl2tc new_code_decl(lhs->type, identifier);
    c_instructiont new_decl_instr(DECL);
    new_decl_instr.code = new_code_decl;
    new_decl_instr.location = func_call.location;
    res.push_back(new_decl_instr);
    // Assignment
    expr2tc oper = code_function_call->operands[i];
    code_assign2tc new_code_assign(lhs, oper);
    c_instructiont new_assign_instr(ASSIGN);
    new_assign_instr.code = new_code_assign;
    new_assign_instr.location = func_call.location;
    res.push_back(new_assign_instr);
  }
  return res;
}

void inline_function_calls(
  goto_functionst goto_functions)
{
  unsigned int scope_id_counter = 0;
  std::vector<unsigned int> scope_stack;
  scope_stack.push_back(scope_id_counter);
  for(unsigned int i = 0; i < instructions_to_c.size(); i++)
  {
    c_instructiont instr = instructions_to_c.at(i);
    instructions_to_c.at(i).scope_id = scope_stack.back();
    //std::cerr << ">>>>> instr_scope_id = " << instructions_to_c.at(i).scope_id << "\n";
    if(instr.type == FUNCTION_CALL)
    {
      // Getting function name symbol from the function call
      assert(is_code_function_call2t(instr.code));
      code_function_call2tc code_function_call =
        to_code_function_call2t(instr.code);
      symbol2t func_call_sym = to_symbol2t(code_function_call->function);
      // Checking if the function exists in the symbol map and it has a body
      auto function = goto_functions.function_map.find(func_call_sym.thename);
      if(function != goto_functions.function_map.end() && function->second.body_available)
      {
        // We are entering a new scope here
        scope_id_counter++;
        scope_stack.push_back(scope_id_counter);
        // Create a SKIP instruction with the scope_begin flag set
        c_instructiont scope_instr(SKIP);
        scope_instr.scope_begin = true;
        scope_instr.location = instr.location;
        scope_instr.scope_id = scope_id_counter;
        i++;
        instructions_to_c.insert(instructions_to_c.begin() + i, scope_instr);
        std::vector<c_instructiont> fun_instr =
          inline_function_call(instr, function->second);
        for(auto it = fun_instr.begin(); it != fun_instr.end(); it++)
        {
          i++;
          it->scope_id = scope_id_counter;
          instructions_to_c.insert(instructions_to_c.begin() + i, *it);
        }
      }
    }
    else if(instr.type == END_FUNCTION)
    {
      // Create a SKIP instruction with the scope_begin flag set
      c_instructiont scope_instr(SKIP);
      scope_instr.scope_end = true;
      scope_instr.location = instr.location;
      scope_instr.scope_id = scope_stack.back();
      instructions_to_c.insert(instructions_to_c.begin() + i, scope_instr);
      i++;
      // We are leaving the scope int here. So pop the current scope from
      // the scope stack
      scope_stack.pop_back();
    }
  }
}

void insert_static_declarations(namespacet ns)
{
  // Declaring all extern variables but not functions
  ns.get_context().foreach_operand_in_order(
    [&ns](const symbolt &s) {
      if(s.is_extern && s.id != "argv\'" && 
          s.id != "argc\'" && s.type.id() != "code")
      {
        code_declt decl(symbol_expr(s));
        code_decl2tc code_decl;
        migrate_expr(decl, code_decl);
        c_instructiont instr(DECL);
        //instr.decl = decl;
        instr.code = code_decl;
        instr.location = s.location;
        instructions_to_c.insert(instructions_to_c.begin(), instr);
      }

      /*
      if(s.id == "c:@__ESBMC_alloc_size")
      {
        //std::cerr << ">>>>> s.id = " << s.id << "\n";
        std::cerr << ">>>>> s = " << s << "\n";
      }
      */
    });
  // Turning the initialisations in the form of assignments only
  // for variables with static lifetime into the initialisations
  // in the form of declarations with initialisers
  for(unsigned int i = 0; i < instructions_to_c.size(); i++)
  {
    c_instructiont instr = instructions_to_c.at(i);
    if(instr.type == ASSIGN)
    {
      assert(is_code_assign2t(instr.code));
      code_assign2t assign = to_code_assign2t(instr.code);
      symbol2tc sym = get_symbol(assign.target);
 
      // Skip all the assignments for argv and argc
      if(sym->thename == "argv\'" ||
         sym->thename == "argc\'")
      {
        instructions_to_c.erase(instructions_to_c.begin() + i);
        i--;
        continue;
      }

      code_decl2tc code_decl(sym->type, sym->thename);
      c_instructiont decl_instr(DECL);
      decl_instr.code = code_decl;
      decl_instr.location = instr.location;

      instructions_to_c.insert(instructions_to_c.begin() + i, decl_instr);
      i++;
    }
    // The globals initialisation block has finished
    if(
      instr.type == FUNCTION_CALL &&
      get_function_name(instr.code) == "pthread_start_main_hook")
      break;
  }
}

void assign_returns()
{
  std::vector<code_function_call2t> fun_call_list;
  for(unsigned int i = 0; i < instructions_to_c.size(); i++)
  {
    c_instructiont instr = instructions_to_c.at(i);
    if(instr.type == FUNCTION_CALL)
    {
      assert(is_code_function_call2t(instr.code));
      code_function_call2t code_function_call =
        to_code_function_call2t(instr.code);
      fun_call_list.push_back(code_function_call);
    }
    if(instr.type == END_FUNCTION)
    {
      fun_call_list.pop_back();
    }
    if(instr.type == RETURN)
    {
      assert(is_code_return2t(instr.code));
      code_function_call2t cur_fun_call = fun_call_list.back();
      if(cur_fun_call.ret != NULL)
      {
        symbol2tc ret_sym = to_symbol2t(cur_fun_call.ret);
        code_return2t code_ret = to_code_return2t(instr.code);
        c_instructiont c_instr(ASSIGN);
        code_assign2tc assign(ret_sym, code_ret.operand);
        c_instr.code = assign;
        instructions_to_c.at(i).code = assign;
      }
    }
  }
  // Making sure that there is END_FUNCTION for every FUNCTION_CALL
  assert(fun_call_list.size() == 0);
}

void merge_decl_assign_pairs()
{
  for(unsigned int i = 0; i < instructions_to_c.size() - 1; i++)
  {
    c_instructiont cur_instr = instructions_to_c.at(i);
    c_instructiont next_instr = instructions_to_c.at(i + 1);
    if(cur_instr.type == DECL && next_instr.type == ASSIGN)
    {
      assert(is_code_decl2t(cur_instr.code));
      assert(is_code_assign2t(next_instr.code));
      code_decl2tc decl = to_code_decl2t(cur_instr.code);
      code_assign2tc assign = to_code_assign2t(next_instr.code);
      assert(is_symbol2t(assign->target));
      if(decl->value == to_symbol2t(assign->target).thename)
      {
        exprt sym_expr = migrate_expr_back(assign->target);
        exprt val_expr = migrate_expr_back(assign->source);
        code_declt decl(sym_expr, val_expr);

        instructions_to_c.at(i).decl = decl;
        instructions_to_c.erase(instructions_to_c.begin() + i + 1);
      }
    }
  }
}

void assign_dynamic_sizes()
{
  unsigned int dyn_size_counter = 0;
  for(unsigned int i = 0; i < instructions_to_c.size(); i++)
  {
    c_instructiont cur_instr = instructions_to_c.at(i);
    if(cur_instr.type == ASSIGN)
    {
      assert(is_code_assign2t(cur_instr.code));
      code_assign2tc assign = to_code_assign2t(cur_instr.code);
      if(is_dynamic_size2t(assign->target))
      {
        dynamic_size2tc dyn_size = to_dynamic_size2t(assign->target);
        assert(is_address_of2t(dyn_size.value));
        // Erase the current instruction "dynamic_size(&arr) = val"
        instructions_to_c.erase(instructions_to_c.begin() + i);
        // Creating a new dynamic_size symbol for every "dynamic_size(&arr) = val" call
        symbol2tc dyn_size_sym(get_uint32_type(),
                            "__ESBMC_dynamic_size_" + std::to_string(dyn_size_counter));
        // Declaring the dynamic_size symbol first
        c_instructiont decl_instr(DECL);
        code_decl2tc decl(dyn_size_sym->type, dyn_size_sym->thename); 
        decl_instr.code = decl;
        decl_instr.location = cur_instr.location;
        instructions_to_c.insert(instructions_to_c.begin() + i, decl_instr);
        i++;
        // Assigning the value to the dynamic_size symbol
        c_instructiont assign_instr(ASSIGN);
        code_assign2tc new_assign(dyn_size_sym, assign->source);
        assign_instr.code = new_assign;
        assign_instr.location = cur_instr.location;
        instructions_to_c.insert(instructions_to_c.begin() + i, assign_instr);
        i++;
        // Adding the dynamic_size symbol id to the map
        dyn_size_map[assign->target] = dyn_size_counter;
        dyn_size_counter++;
      }
      else if(is_union_type(assign->target->type) && 
                is_constant_union2t(assign->source))
      {
        // This is a hack for now
        constant_union2tc const_union = to_constant_union2t(assign->source);
        if(const_union->datatype_members.size() == 1)
        {
          typecast2tc new_source(const_union->type, const_union->datatype_members[0]);
          assign->source = new_source; 
          c_instructiont assign_instr(ASSIGN);
          assign_instr.code = assign;
          assign_instr.location = cur_instr.location;
          instructions_to_c.at(i) = assign_instr;
        }
      }
      else if(is_typecast2t(assign->target))
      {
        // Creating a new instruction where the LHS typecast is removed
        assign->target = get_base_expr(assign->target); 
        c_instructiont assign_instr(ASSIGN);
        assign_instr.code = assign;
        assign_instr.location = cur_instr.location;
        instructions_to_c.at(i) = assign_instr;
      }
    }
  }
}

void output_typedefs(namespacet &ns, std::ostream &out)
{
  // Declaring extern variables first
  ns.get_context().foreach_operand_in_order(
    [&ns, &out](const symbolt &s) {
      std::smatch m;
      //std::string tag = s.type.tag().as_string();
      std::string tag = s.id.as_string();
      if(std::regex_search(tag, m, std::regex("tag\\-.+")))
      {
        //std::cerr << ">>>>> found a tag: " << s.id << "\n";
        //if(!(std::regex_search(tag, m, std::regex("tag\\-struct.+")) || 
        //      std::regex_search(tag, m, std::regex("tag\\-union.+"))))
        {
          out << type2ccode(s.type, ns) << ";\n";
        }
      }
    });
}

void output_execution_trace(namespacet &ns, std::ostream &out)
{
  out << "void run_execution_trace(int argc, char *argv[])\n";
  out << "{\n";

  out << "// Outputting typedefs\n";
  output_typedefs(ns, out);

  out << "// Outputting the rest of the instructions\n";
  // Converting the instructions to C
  for(c_instructiont it : instructions_to_c)
    out << it.convert_to_c(ns) << "\n";

  out << "}\n";
}

