#include <goto-programs/forward_slicer.h>

namespace
{
// Recursively try to extract symbols from an expression
void get_symbols(const expr2tc &expr, std::set<std::string> &values)
{
  // TODO: This function should return a set!
  if(!expr)
    return;
  switch(expr->expr_id)
  {
  case expr2t::symbol_id:

    values.insert(to_symbol2t(expr).get_symbol_name());
    return;

  case expr2t::dynamic_size_id:
    get_symbols(to_dynamic_size2t(expr).value, values);
    return;

  case expr2t::address_of_id:
    get_symbols(to_address_of2t(expr).ptr_obj, values);
    return;    

  case expr2t::index_id:
    get_symbols(to_index2t(expr).source_value, values);
    get_symbols(to_index2t(expr).index, values);
    return;

  case expr2t::with_id:
    return get_symbols(to_with2t(expr).update_value, values);

  case expr2t::byte_extract_id:
    return get_symbols(to_byte_extract2t(expr).source_value, values);

  case expr2t::typecast_id:
    return get_symbols(to_typecast2t(expr).from, values);

  case expr2t::bitcast_id:
    return get_symbols(to_bitcast2t(expr).from, values);

  case expr2t::code_assign_id:
    return get_symbols(to_code_assign2t(expr).target, values);

  case expr2t::code_decl_id:
    values.insert(to_code_decl2t(expr).value.as_string());
    return;

  case expr2t::code_dead_id:
    values.insert(to_code_dead2t(expr).value.as_string());
    return;

  case expr2t::code_return_id:
    to_code_return2t(expr).foreach_operand(
      [&values](auto &e) { get_symbols(e, values); });
    return;

  case expr2t::code_function_call_id:
    to_code_function_call2t(expr).foreach_operand(
      [&values](auto &e) { get_symbols(e, values); });
    return;

  case expr2t::constant_struct_id:
    for(const auto &v : to_constant_struct2t(expr).datatype_members)
      get_symbols(v, values);
    return;

  case expr2t::constant_union_id:
    for(const auto &v : to_constant_union2t(expr).datatype_members)
      get_symbols(v, values);
    return;
  case expr2t::member_id:
    get_symbols(to_member2t(expr).source_value, values);
    return;

  case expr2t::if_id:
  {
    get_symbols(to_if2t(expr).true_value, values);
    get_symbols(to_if2t(expr).false_value, values);
    return;
  }

  case expr2t::not_id:
  {
    get_symbols(to_not2t(expr).value, values);
    return;
  }

  case expr2t::bswap_id:
  {
    get_symbols(to_bswap2t(expr).value, values);
    return;
  }

  case expr2t::ieee_add_id:
  case expr2t::ieee_div_id:
  case expr2t::ieee_mul_id:
  case expr2t::ieee_sub_id:
  {
    auto op = std::dynamic_pointer_cast<ieee_arith_2ops>(expr);
    get_symbols(op->side_1, values);
    get_symbols(op->side_2, values);
    return;
  }

  case expr2t::add_id:
  case expr2t::div_id:
  case expr2t::mul_id:
  case expr2t::sub_id:
  {
    auto op = std::dynamic_pointer_cast<arith_2ops>(expr);
    get_symbols(op->side_1, values);
    get_symbols(op->side_2, values);
    return;
  }

  case expr2t::and_id:
  case expr2t::or_id:
  case expr2t::xor_id:
  {
    auto op = std::dynamic_pointer_cast<logic_2ops>(expr);
    get_symbols(op->side_1, values);
    get_symbols(op->side_2, values);
    return;
  }

  case expr2t::lessthan_id:
  case expr2t::lessthanequal_id:
  case expr2t::greaterthan_id:
  case expr2t::greaterthanequal_id:
  case expr2t::equality_id:
  case expr2t::notequal_id:
  {
    auto op = std::dynamic_pointer_cast<relation_data>(expr);
    get_symbols(op->side_1, values);
    get_symbols(op->side_2, values);
    return;
  }

  case expr2t::signbit_id:
  case expr2t::popcount_id:
  {
    auto op = std::dynamic_pointer_cast<overflow_ops>(expr);
    get_symbols(op->operand, values);
    return;
  }

  case expr2t::bitand_id:
  case expr2t::bitor_id:
  case expr2t::bitxor_id:
  case expr2t::bitnand_id:
  case expr2t::bitnor_id:
  case expr2t::bitnxor_id:
  case expr2t::lshr_id:
  case expr2t::shl_id:
  case expr2t::ashr_id:
  case expr2t::concat_id:
  {
    auto op = std::dynamic_pointer_cast<bit_2ops>(expr);
    get_symbols(op->side_1, values);
    get_symbols(op->side_2, values);
    return;
  }

  case expr2t::constant_int_id:
  case expr2t::constant_floatbv_id:
  case expr2t::constant_bool_id:
  case expr2t::constant_array_of_id:
  case expr2t::sideeffect_id:
    // TODO
    return;
  default:
    log_error("Missing operator {}", *expr);
    throw "TODO: Implement missing operators";
    return;
  }
}
} // namespace

void slicer_domaint::output(std::ostream &out) const
{
  if(is_bottom())
  {
    out << "BOTTOM\n";
    return;
  }

  out << "\t[ ";
  for(const auto &d : dependencies)
  {
    out << "\n\t\t" << d.first << ": {";
    for(const auto &s : d.second)
    {
      out << " " << s << " ";
    }
    out << d.first << " }\n\t";
  }

  out << "]";
}

void slicer_domaint::assign(const expr2tc &e)
{
  auto assignment = to_code_assign2t(e);

  // We don't care about constants
  if(!is_symbol2t(assignment.target))
    return;

  // We do not support pointers
  if(is_pointer_type(assignment.target->type))
  {
    log_error("Forward slicer does not support pointers");
    throw "TODO: points-to analysis";
  }
  
  std::set<std::string> vars;
  get_symbols(assignment.source, vars);
  // Some magic to get the vars
  dependencies[to_symbol2t(assignment.target).get_symbol_name()] = vars;
}

void slicer_domaint::declaration(const expr2tc &e)
{
  auto A = to_code_decl2t(e);
  if(is_pointer_type(A.type))
  {
    log_error("Forward slicer does not support pointers");
    throw "TODO: points-to analysis";
  }

  std::set<std::string> vars;
  // Some magic to get the vars
  dependencies[A.value.as_string()] = vars;
}

void slicer_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett,
  ai_baset &,
  const namespacet &ns)
{
  (void)ns;

  const goto_programt::instructiont &instruction = *from;
  switch(instruction.type)
  {
  case DECL:
    declaration(instruction.code);
    break;

  case ASSIGN:
    assign(instruction.code);
    break;

  default:;
  }
}

bool slicer_domaint::join(const slicer_domaint &b)
{
  // If old domain is empty, do nothing
  if(b.is_bottom())
    return false;

  bool changed = false;
  // First, merge new elements
  for(auto &d : b.dependencies)
  {
    if(!dependencies.count(d.first))
    {
      dependencies[d.first] = d.second;
      changed = true;
    }
    else
      for(auto &x : d.second)
        changed |= dependencies[d.first].insert(x).second;
  }

  if(!changed)
    return false;

  /* Find a fixpoint
   * This will always terminate since:
   *   - The domain is a finite set
   *   - The meet is the union
   *
  */
  while(changed)
  {
    changed = false;
    for(auto &d : dependencies)
      for(auto &item : d.second)
        for(auto &item_2 : dependencies[item])
          changed |= dependencies[d.first].insert(item_2).second;
  }

  return true;
}

bool forward_slicer::runOnProgram(goto_functionst &F)
{
  sl(F, ns);
  return true;
}

bool forward_slicer::runOnLoop(loopst &loop, goto_programt &)
{
  // Let's remove empty loops
  goto_programt::targett loop_head = loop.get_original_loop_head();
  while((++loop_head)->is_skip())
    ;
  if(loop_head == loop.get_original_loop_exit())
  {
    loop.get_original_loop_head()->make_skip();
    loop.get_original_loop_exit()->make_skip();
    return true;
  }
  return false;
}
#include <util/prefix.h>
bool forward_slicer::should_skip_function(const std::string &func)
{
  return has_prefix(func, "c:@F@__ESBMC_") || has_prefix(func, "c:@F@pthread_");
}

bool forward_slicer::runOnFunction(std::pair<const dstring, goto_functiont> &F)
{
  if(should_skip_function(F.first.as_string()))
    return false;
  if(F.second.body_available)
  {
    std::set<std::string> deps;
    for(auto it = (F.second.body).instructions.rbegin();
        it != (F.second.body).instructions.rend();
        ++it)
    {
      switch(it->type)
      {
      case ASSUME:
      case ASSERT:
      case FUNCTION_CALL:
      case RETURN:
      case GOTO:
      {
        get_symbols(it->guard, deps);
        get_symbols(it->code, deps);
        auto base = it;
        base++;
        auto state = sl[base.base()].dependencies;
        for(auto &d : deps)
          for(auto &item : state[d])
            deps.insert(item);
        break;
      }
      default:
        break;
      }
    }
    for(auto it = (F.second.body).instructions.rbegin();
        it != (F.second.body).instructions.rend();
        ++it)
    {
      switch(it->type)
      {
      case DECL:
      case DEAD:
      case ASSUME:
      case ASSERT:
      case ASSIGN:
      {
        std::set<std::string> local_deps;
        get_symbols(it->code, local_deps);
        bool contains_dep = false;
        for(auto &item : local_deps)
        {
          if(deps.count(item))
          {
            contains_dep = true;
            break;
          }
        }
        if(!contains_dep)
          it->make_skip();
        break;
      }

      default:;
        break;
      }
    }
  }
  return true;
}
