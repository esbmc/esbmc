#include <goto-programs/goto_contractor.h>

void goto_contractor(goto_functionst &goto_functions)
{
  goto_contractort gotoContractort(goto_functions);

  goto_functions.update();
}

void goto_contractort::get_contractors(goto_functionst goto_functions)
{
  auto function = goto_functions.function_map.find("c:@F@main");
  for(const auto &ins : function->second.body.instructions)
  {
    if(ins.is_assert())
      contractors.add_contractor(
        create_contractor_from_expr2t(ins.guard), ins.location_number);
    else if(
      ins.is_function_call() &&
      is_symbol2t(to_code_function_call2t(ins.code).function) &&
      to_symbol2t(to_code_function_call2t(ins.code).function)
          .get_symbol_name() == "c:@F@__VERIFIER_assert")
      contractors.add_contractor(
        create_contractor_from_expr2t(
          to_code_function_call2t(ins.code).operands[0]),
        ins.location_number);
  }
}

void goto_contractort::get_intervals(goto_functionst goto_functions)
{
  auto function = goto_functions.function_map.find("c:@F@main");

  for(const auto &ins : function->second.body.instructions)
    if(ins.is_assume())
      parse_intervals(ins.guard);
}

void goto_contractort::parse_intervals(expr2tc expr)
{
  symbol2tc symbol;
  double value;

  expr = get_base_object(expr);

  if(is_and2t(expr))
  {
    parse_intervals(to_and2t(expr).side_1);
    parse_intervals(to_and2t(expr).side_2);
    return;
  }

  if(!is_comp_expr(expr))
    return;

  std::shared_ptr<relation_data> rel;
  rel = std::dynamic_pointer_cast<relation_data>(expr);

  //side1 1 is a symbol or typecast to symbol
  auto side1 = rel->side_1;
  //side 2 is always a number.
  auto side2 = rel->side_2;

  auto obj = get_base_object(side1);
  if(!is_symbol2t(obj))
    return;

  symbol = to_symbol2t(obj);
  bool neg = false;
  if(is_neg2t(side2))
  {
    neg = true;
    side2 = to_neg2t(side2).value;
  }

  if(!is_constant_int2t(side2))
    return;

  value = to_constant_int2t(side2).as_long() * (neg ? -1 : 1) + (neg ? -1 : 1);

  if(map.find(symbol->get_symbol_name()) == CspMap::NOT_FOUND)
    return;
  switch(expr->expr_id)
  {
  case expr2t::expr_ids::greaterthan_id:
  case expr2t::expr_ids::greaterthanequal_id:
    map.update_lb_interval(value, symbol->get_symbol_name());
    break;
  case expr2t::expr_ids::lessthan_id:
  case expr2t::expr_ids::lessthanequal_id:
    map.update_ub_interval(value, symbol->get_symbol_name());
    break;
  case expr2t::expr_ids::equality_id:
    break;
  default:;
  }
}

void goto_contractort::insert_assume(goto_functionst goto_functions)
{
  loopst loop;
  unsigned int last_loc = 0;

  ///This loop is to find the last loop in the code based on location
  for(auto &function_loop : function_loops)
    if(last_loc < function_loop.get_original_loop_head()->location_number)
    {
      loop = function_loop;
      last_loc = loop.get_original_loop_head()->location_number;
    }

  auto loop_exit = loop.get_original_loop_exit();

  goto_programt dest;

  auto goto_function = goto_functions.function_map.find("c:@F@main")->second;

  if(map.is_empty_set())
  {
    auto cond = gen_zero(int_type2());
    goto_programt tmp_e;
    auto e = tmp_e.add_instruction(ASSUME);
    e->inductive_step_instruction = false;
    e->guard = cond;
    e->location = loop_exit->location;
    goto_function.body.destructive_insert(loop_exit, tmp_e);
  }
  else
    for(auto const &var : map.var_map)
    {
      //testing with updating both bounds after reduction.
      symbol2tc X = var.second.getSymbol();
      if(var.second.isIntervalChanged())
      {
        auto ub = create_value_expr(var.second.getInterval().ub(), int_type2());
        auto cond2 = create_lessthanequal_relation(X, ub);
        goto_programt tmp_e2;
        auto e2 = tmp_e2.add_instruction(ASSUME);
        e2->inductive_step_instruction = false;
        e2->guard = cond2;
        e2->location = loop_exit->location;
        goto_function.body.destructive_insert(loop_exit, tmp_e2);
      }
    }
}

void goto_contractort::apply_contractor()
{
  Contractor *contractor = contractors.get_contractors();

  //Take the intersection of all contractors to perform Algorithm 2
  ibex::CtcFixPoint c_out(*contractor->get_outer());
  //Take the union of all contractors with complement constraints to perform Algorithm 3
  ibex::CtcFixPoint c_in(*contractor->get_inner());

  std::ostringstream oss;

  domains = map.create_interval_vector();

  oss << "Contractors applied:";
  oss << "\n\t- Domains (before): " << domains;
  auto X = domains;

  c_in.contract(X);

  oss << "\n\t- Domains (after): " << X;
  map.update_intervals(X);

  log_status("{}", oss.str());
}

ibex::Ctc *goto_contractort::create_contractor_from_expr2t(const expr2tc &expr)
{
  ibex::Ctc *contractor = nullptr;
  auto base_object = get_base_object(expr);

  if(is_constraint_operator(expr))
  {
    auto cons = create_constraint_from_expr2t(base_object);
    if(cons == nullptr)
      return nullptr;
    contractor = new ibex::CtcFwdBwd(*cons);
  }
  else
  {
    std::shared_ptr<logic_2ops> logic_op;
    logic_op = std::dynamic_pointer_cast<logic_2ops>(base_object);

    switch(base_object->expr_id)
    {
    case expr2t::expr_ids::and_id:
    {
      auto side1 = create_contractor_from_expr2t(logic_op->side_1);
      auto side2 = create_contractor_from_expr2t(logic_op->side_2);
      if(side1 != nullptr && side2 != nullptr)
        contractor = new ibex::CtcCompo(*side1, *side2);
      break;
    }
    case expr2t::expr_ids::or_id:
    {
      auto side1 = create_contractor_from_expr2t(logic_op->side_1);
      auto side2 = create_contractor_from_expr2t(logic_op->side_2);
      if(side1 != nullptr && side2 != nullptr)
        contractor = new ibex::CtcUnion(*side1, *side2);
      break;
    }
    case expr2t::expr_ids::notequal_id:
    {
      std::shared_ptr<relation_data> rel;
      rel = std::dynamic_pointer_cast<relation_data>(base_object);
      ibex::Function *f = create_function_from_expr2t(rel->side_1);
      ibex::Function *g = create_function_from_expr2t(rel->side_2);
      if(g == nullptr || f == nullptr)
        return nullptr;
      auto *side1 = new ibex::NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
      auto *side2 = new ibex::NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
      auto *c_side1 = new ibex::CtcFwdBwd(*side1);
      auto *c_side2 = new ibex::CtcFwdBwd(*side2);
      contractor = new ibex::CtcUnion(*c_side1, *c_side2);
      break;
    }
    case expr2t::expr_ids::not_id:
      //TODO: implement "not" flag  and process.
      parse_error(base_object);
      break;
    default:
      parse_error(base_object);
      break;
    }
  }
  return contractor;
}

bool goto_contractort::is_constraint_operator(const expr2tc &expr)
{
  expr2tc e = get_base_object(expr);
  return (is_comp_expr(e) && !is_notequal2t(e)) || is_arith_expr(e);
}

ibex::NumConstraint *
goto_contractort::create_constraint_from_expr2t(const expr2tc &expr)
{
  ibex::NumConstraint *c;

  if(is_unsupported_operator_in_constraint(expr))
  {
    parse_error(expr);
    return nullptr;
  }
  if(is_not2t(expr))
  {
    //TODO: implement "not" flag  and process.
    parse_error(expr);
    return nullptr;
  }

  std::shared_ptr<relation_data> rel;
  rel = std::dynamic_pointer_cast<relation_data>(get_base_object(expr));

  ibex::Function *f, *g;
  f = create_function_from_expr2t(rel->side_1);
  g = create_function_from_expr2t(rel->side_2);
  if(f == nullptr || g == nullptr)
    return nullptr;
  //TODO: implement "not" flag  and process.
  switch(get_base_object(expr)->expr_id)
  {
  case expr2t::expr_ids::greaterthanequal_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) >= (*g)(*vars));
    break;
  case expr2t::expr_ids::greaterthan_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
    break;
  case expr2t::expr_ids::lessthanequal_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) <= (*g)(*vars));
    break;
  case expr2t::expr_ids::lessthan_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
    break;
  case expr2t::expr_ids::equality_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) = (*g)(*vars));
    break;
  default:
    return nullptr;
  }
  return c;
}

bool goto_contractort::is_unsupported_operator_in_constraint(
  const expr2tc &expr)
{
  expr2tc e = get_base_object(expr);
  return is_arith_expr(e) || is_constant_number(e) || is_symbol2t(e) ||
         is_notequal2t(e) || is_not2t(e) || is_modulus2t(e) || is_or2t(e) ||
         is_and2t(e);
}

ibex::Function *goto_contractort::create_function_from_expr2t(expr2tc expr)
{
  ibex::Function *f = nullptr;
  ibex::Function *g, *h;

  if(is_comp_expr(expr))
  {
    parse_error(expr);
    return nullptr;
  }
  switch(expr->expr_id)
  {
  case expr2t::expr_ids::add_id:
  case expr2t::expr_ids::sub_id:
  case expr2t::expr_ids::mul_id:
  case expr2t::expr_ids::div_id:
  {
    std::shared_ptr<arith_2ops> arith_op;
    if(!is_arith_expr(expr))
      return nullptr;
    arith_op = std::dynamic_pointer_cast<arith_2ops>(expr);
    g = create_function_from_expr2t(arith_op->side_1);
    h = create_function_from_expr2t(arith_op->side_2);
    if(g == nullptr || h == nullptr)
      return nullptr;

    switch(arith_op->expr_id)
    {
    case expr2t::expr_ids::add_id:
      f = new ibex::Function(*vars, (*g)(*vars) + (*h)(*vars));
      break;
    case expr2t::expr_ids::sub_id:
      f = new ibex::Function(*vars, (*g)(*vars) - (*h)(*vars));
      break;
    case expr2t::expr_ids::mul_id:
      f = new ibex::Function(*vars, (*g)(*vars) * (*h)(*vars));
      break;
    case expr2t::expr_ids::div_id:
      f = new ibex::Function(*vars, (*g)(*vars) / (*h)(*vars));
      break;
    default:
      return nullptr;
    }
    break;
  }
  case expr2t::expr_ids::neg_id:
  {
    h = create_function_from_expr2t(to_neg2t(expr).value);
    f = new ibex::Function(*vars, -(*h)(*vars));
    break;
  }
  case expr2t::expr_ids::symbol_id:
  {
    int index = create_variable_from_expr2t(expr);
    if(index != CspMap::NOT_FOUND)
    {
      f = new ibex::Function(*vars, (*vars)[index]);
    }
    else
    {
      log_error("ERROR: MAX VAR SIZE REACHED");
      return nullptr;
    }
    break;
  }
  case expr2t::expr_ids::typecast_id:
    f = create_function_from_expr2t(to_typecast2t(expr).from);
    break;
  case expr2t::expr_ids::constant_int_id:
  {
    const ibex::ExprConstant &c =
      ibex::ExprConstant::new_scalar(to_constant_int2t(expr).value.to_int64());
    f = new ibex::Function(*vars, c);
    break;
  }
  default:
    f = nullptr;
  }

  return f;
}

int goto_contractort::create_variable_from_expr2t(expr2tc expr)
{
  if(is_symbol2t(expr))
  {
    std::string var_name = to_symbol2t(expr).get_symbol_name();
    int index = map.add_var(var_name, to_symbol2t(expr));
    return index;
  }
  return map.NOT_FOUND;
}

void goto_contractort::parse_error(const expr2tc &expr)
{
  std::ostringstream oss;
  oss << get_expr_id(expr);
  oss << " Expression is complex, skipping this assert.\n";
  log_debug("{}", oss.str());
}

bool goto_contractort::initialize_main_function_loops()
{
  auto it = goto_functions.function_map.find("c:@F@main");
  {
    number_of_functions++;
    runOnFunction(*it);
    if(it->second.body_available)
    {
      goto_loopst goto_loops(it->first, goto_functions, it->second);
      this->function_loops = goto_loops.get_loops();
    }
  }
  goto_functions.update();
  return true;
}

const ibex::Interval &vart::getInterval() const
{
  return interval;
}

size_t vart::getIndex() const
{
  return index;
}

vart::vart(const string &varName, const symbol2tc &symbol, const size_t &index)
{
  this->var_name = varName;
  this->symbol = symbol;
  this->index = index;
  interval_changed = false;
}

void vart::setInterval(const ibex::Interval &interval)
{
  this->interval = interval;
}

bool vart::isIntervalChanged() const
{
  return interval_changed;
}

void vart::setIntervalChanged(bool intervalChanged)
{
  interval_changed = intervalChanged;
}

const symbol2tc &vart::getSymbol() const
{
  return symbol;
}
