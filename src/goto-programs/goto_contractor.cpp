#include <goto-programs/goto_contractor.h>
#include <goto-programs/abstract-interpretation/interval_domain.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>

void goto_contractor(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options)
{
  goto_contractort gotoContractort(goto_functions, ns, options);

  goto_functions.update();
}

void goto_contractort::get_contractors(goto_functionst goto_functions)
{
  auto function = goto_functions.function_map.find("c:@F@main");
  for (const auto &ins : function->second.body.instructions)
  {
    if (ins.is_assert())
    {
      contractors.add_contractor(parser.parse(ins.guard), ins.location_number);
    }
    else if (
      ins.is_function_call() &&
      is_symbol2t(to_code_function_call2t(ins.code).function) &&
      to_symbol2t(to_code_function_call2t(ins.code).function)
          .get_symbol_name() == "c:@F@__VERIFIER_assert")
      contractors.add_contractor(
        parser.parse(to_code_function_call2t(ins.code).operands[0]),
        ins.location_number);
  }
}

void goto_contractort::get_intervals(
  goto_functionst goto_functions,
  const namespacet &ns)
{
  ait<interval_domaint> interval_analysis;
  interval_analysis(goto_functions, ns);

  std::ostringstream oss;
  interval_analysis.output(goto_functions, oss);

  auto f_it = goto_functions.function_map.find("c:@F@main");
  {
    Forall_goto_program_instructions (i_it, f_it->second.body)
    {
      if (
        i_it->is_assert() ||
        (i_it->is_function_call() &&
         is_symbol2t(to_code_function_call2t(i_it->code).function) &&
         to_symbol2t(to_code_function_call2t(i_it->code).function)
             .get_symbol_name() == "c:@F@__VERIFIER_assert"))
      {
        auto it = map.var_map.begin();
        while (it != map.var_map.end())
        {
          auto var_name = to_symbol2t(it->second.getSymbol()).get_symbol_name();
          auto new_interval = interval_analysis[i_it].get_int_map()[var_name];
          if (!new_interval.is_top())
          {
            auto lb = new_interval.get_lower().to_int64();
            auto ub = new_interval.get_upper().to_int64();
            auto i = it->second.getInterval();

            oss << "at location: " << i_it->location_number
                << " Var: " << var_name << " this out " << i
                << " should be: " << new_interval << std::endl;

            if (new_interval.lower && (isinf(i.lb()) || i.lb() > lb))
              map.update_lb_interval(lb, var_name);
            if (new_interval.upper && (isinf(i.ub()) || i.ub() < ub))
              map.update_ub_interval(ub, var_name);
          }
          it++;
        }
        log_debug("contractor", "{}", oss.str());
      }
    }
  }
  goto_functions.update();
}

void goto_contractort::parse_intervals(expr2tc expr)
{
  double value;

  expr = get_base_object(expr);

  if (is_and2t(expr))
  {
    parse_intervals(to_and2t(expr).side_1);
    parse_intervals(to_and2t(expr).side_2);
    return;
  }

  if (!is_comp_expr(expr))
    return;

  const relation_data &rel = dynamic_cast<const relation_data &>(*expr);

  //side1 1 is a symbol or typecast to symbol
  auto side1 = rel.side_1;
  //side 2 is always a number.
  auto side2 = rel.side_2;

  auto obj = get_base_object(side1);
  if (!is_symbol2t(obj))
    return;

  bool neg = false;
  if (is_neg2t(side2))
  {
    neg = true;
    side2 = to_neg2t(side2).value;
  }

  if (!is_constant_int2t(side2))
    return;

  value = to_constant_int2t(side2).as_long() * (neg ? -1 : 1) + (neg ? -1 : 1);

  std::string symbol_name = to_symbol2t(obj).get_symbol_name();
  if (map.find(symbol_name) == CspMap::NOT_FOUND)
    return;
  switch (expr->expr_id)
  {
  case expr2t::expr_ids::greaterthan_id:
  case expr2t::expr_ids::greaterthanequal_id:
    map.update_lb_interval(value, symbol_name);
    break;
  case expr2t::expr_ids::lessthan_id:
  case expr2t::expr_ids::lessthanequal_id:
    map.update_ub_interval(value, symbol_name);
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
  for (auto &function_loop : function_loops)
    if (last_loc < function_loop.get_original_loop_head()->location_number)
    {
      loop = function_loop;
      last_loc = loop.get_original_loop_head()->location_number;
    }

  auto loop_exit = loop.get_original_loop_exit();

  goto_programt dest;

  auto goto_function = goto_functions.function_map.find("c:@F@main")->second;

  if (map.is_empty_set())
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
    for (auto const &var : map.var_map)
    {
      //testing with updating both bounds after reduction.
      expr2tc X = var.second.getSymbol();
      if (var.second.isIntervalChanged())
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

bool goto_contractort::initialize_main_function_loops()
{
  auto it = goto_functions.function_map.find("c:@F@main");
  {
    number_of_functions++;
    runOnFunction(*it);
    if (it->second.body_available)
    {
      goto_loopst goto_loops(it->first, goto_functions, it->second);
      this->function_loops = goto_loops.get_loops();
    }
  }
  goto_functions.update();
  return true;
}

void goto_contractort::insert_assume_at(
  goto_functiont goto_function,
  std::_List_iterator<goto_programt::instructiont> instruction)
{
  /// Here we build an assume instruction with a conjunction of multiple conditions.
  /// We start with a true expression and add other conditions with and2tc
  /// eventually, we will have somthing like: true and x>0 and x<10
  expr2tc cond = gen_true_expr();
  for (auto const &var : map.var_map)
  {
    //testing with updating both bounds after reduction.
    auto X = var.second.getSymbol();
    {
      if (var.second.getInterval().is_empty())
        continue;
      if (
        isinf(var.second.getInterval().lb()) &&
        isinf(var.second.getInterval().ub()))
        continue;
      if (!isinf(var.second.getInterval().lb()))
      {
        auto lb = create_value_expr(var.second.getInterval().lb(), int_type2());
        auto cond1 = greaterthanequal2tc(X, lb);
        cond = and2tc(cond, cond1);
      }
      if (!isinf(var.second.getInterval().ub()))
      {
        auto ub = create_value_expr(var.second.getInterval().ub(), int_type2());
        auto cond2 = lessthanequal2tc(X, ub);
        cond = and2tc(cond, cond2);
      }
    }
  }
  // Copy current instruction (apart from the target number)
  // and insert it straight after the current instruction
  goto_programt::instructiont copy_instruction;
  copy_instruction.type = instruction->type;
  copy_instruction.guard = instruction->guard;
  copy_instruction.code = instruction->code;
  copy_instruction.location = instruction->location;
  copy_instruction.targets = instruction->targets;
  goto_function.body.instructions.insert(
    std::next(instruction), copy_instruction);
  // Change current instruction into an ASSUME
  instruction->type = ASSUME;
  instruction->code = expr2tc();
  instruction->guard = cond;
  instruction->targets.clear();
}

void goto_contractort::goto_contractor_condition(
  const namespacet &namespacet,
  const optionst &optionst)
{
  ait<interval_domaint> interval_analysis;

  interval_domaint::set_options(optionst);
  interval_analysis(goto_functions, namespacet);
  std::ostringstream oss;

  Forall_goto_functions (f_it, goto_functions)
  {
    Forall_goto_program_instructions (i_it, f_it->second.body)
    {
      if (i_it->is_goto() && !is_true(i_it->guard)) //if or if-else or loop
      {
        Contractor contractor;

        //create contractor and domains
        if (is_not2t(i_it->guard))
        {
          vars = new ibex::Variable(CspMap::MAX_VAR);
          map = CspMap();
          parser = expr_to_ibex_parser(&map, vars);
          contractor = Contractor(parser.parse(to_not2t(i_it->guard).value), 0);

          auto out = contractor.get_outer();
          if (out == nullptr)
          {
            continue;
          }

          interval_analysis(goto_functions, namespacet);
          auto interval_map = interval_analysis[i_it].get_int_map();
          auto it = interval_map.begin();

          while (it != interval_map.end())
          {
            if (it->second.lower)
              map.update_lb_interval(
                it->second.get_lower().to_int64(), it->first.as_string());
            if (it->second.upper)
              map.update_ub_interval(
                it->second.get_upper().to_int64(), it->first.as_string());
            it++;
          }

          auto in = contractor.get_inner();
          auto X_in = map.create_interval_vector();
          auto X_out = map.create_interval_vector();

          out->contract(X_out);

          in->contract(X_in);

          map.update_intervals(X_out);

          auto goto_target = i_it->get_target();
          goto_target--;

          if (goto_target->is_goto()) //tarrget-1 is goto and
          {
            if (!goto_target->is_backwards_goto())
            {
              // IF-ELSE
              //ELSE clause gets the inner contractor results
              auto next = std::next(i_it);
              insert_assume_at(f_it->second, next);
              map.update_intervals(X_in);
              goto_target++;

              insert_assume_at(f_it->second, goto_target);
            }
            else
            {
              // LOOP
              //TODO: Add a check if loop is monotonic
            }
          }
          else
          {
            // IF-THEN
            auto next = i_it;
            next++;

            insert_assume_at(f_it->second, next);
          }
          delete (vars);
        }
      }
      if (
        i_it->is_assert() ||
        (i_it->is_function_call() &&
         is_symbol2t(to_code_function_call2t(i_it->code).function) &&
         to_symbol2t(to_code_function_call2t(i_it->code).function)
             .get_symbol_name() == "c:@F@__VERIFIER_assert"))
      {
        //convert map to ibex
        //first create the contractor to populate cspmap with the variables.
        vars = new ibex::Variable(CspMap::MAX_VAR);
        map = CspMap();
        parser = expr_to_ibex_parser(&map, vars);
        Contractor contractor(parser.parse(i_it->guard), 0);
        auto out = contractor.get_outer();
        if (out == nullptr)
        {
          continue;
        }

        //get intervals and convert them to ibex intervals by updating the map
        interval_analysis(goto_functions, namespacet);
        auto interval_map = interval_analysis[i_it].get_int_map();
        auto it = interval_map.begin();
        while (it != interval_map.end())
        {
          if (it->second.lower)
            map.update_lb_interval(
              it->second.get_lower().to_int64(), it->first.as_string());
          if (it->second.upper)
            map.update_ub_interval(
              it->second.get_upper().to_int64(), it->first.as_string());
          it++;
        }
        auto X = map.create_interval_vector();

        out->contract(X);

        map.update_intervals(X);
        auto next = i_it;
        next++;

        insert_assume_at(f_it->second, next);
        delete (vars);
      }
    }
  }
}

const ibex::Interval &vart::getInterval() const
{
  return interval;
}

size_t vart::getIndex() const
{
  return index;
}

vart::vart(const string &varName, const expr2tc &symbol, const size_t &index)
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

const expr2tc &vart::getSymbol() const
{
  return symbol;
}

vart::vart()
{
}

void vart::dump()
{
  std::ostringstream oss;
  oss << this->var_name << " " << this->getInterval();
  log_status("{}", oss.str());
}

//----------------------------------------------------------------------------------------
ibex::Ctc *
expr_to_ibex_parser::create_contractor_from_expr2t(const expr2tc &expr)
{
  ibex::Ctc *contractor = nullptr;
  expr2tc base_object = get_base_object(expr);

  if (is_not2t(expr))
  {
    auto not_ = to_not2t(base_object);
    contractor = create_contractor_from_expr2t_not(not_.value);
  }
  else if (is_constraint_operator(expr))
  {
    auto cons = create_constraint_from_expr2t(base_object);
    if (cons == nullptr)
      return nullptr;
    contractor = new ibex::CtcFwdBwd(*cons);
    vector_ctc.push_back(contractor);
  }
  else
  {
    switch (base_object->expr_id)
    {
    case expr2t::expr_ids::and_id:
    {
      auto logic_op = to_and2t(base_object);
      auto side1 = create_contractor_from_expr2t(logic_op.side_1);
      auto side2 = create_contractor_from_expr2t(logic_op.side_2);
      if (side1 != nullptr && side2 != nullptr)
      {
        contractor = new ibex::CtcCompo(*side1, *side2);
        vector_ctc.push_back(contractor);
      }
      break;
    }
    case expr2t::expr_ids::or_id:
    {
      auto logic_op = to_or2t(base_object);
      auto side1 = create_contractor_from_expr2t(logic_op.side_1);
      auto side2 = create_contractor_from_expr2t(logic_op.side_2);
      if (side1 != nullptr && side2 != nullptr)
      {
        contractor = new ibex::CtcUnion(*side1, *side2);
        vector_ctc.push_back(contractor);
      }
      break;
    }
    case expr2t::expr_ids::notequal_id:
    {
      //std::shared_ptr<relation_data> rel;
      auto rel = to_notequal2t(base_object);
      ibex::Function *f = create_function_from_expr2t(rel.side_1);
      ibex::Function *g = create_function_from_expr2t(rel.side_2);
      if (g == nullptr || f == nullptr)
        return nullptr;

      auto *side1 = new ibex::NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
      auto *side2 = new ibex::NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
      auto *c_side1 = new ibex::CtcFwdBwd(*side1);
      auto *c_side2 = new ibex::CtcFwdBwd(*side2);
      contractor = new ibex::CtcUnion(*c_side1, *c_side2);

      //for clean up
      vector_nc.push_back(side1);
      vector_nc.push_back(side2);
      vector_ctc.push_back(c_side1);
      vector_ctc.push_back(c_side2);
      vector_ctc.push_back(contractor);
      break;
    }
    case expr2t::expr_ids::typecast_id:
    {
      contractor = create_contractor_from_expr2t(get_base_object(base_object));
      break;
    }
    default:
      parse_error(base_object);
      break;
    }
  }
  return contractor;
}

ibex::Ctc *
expr_to_ibex_parser::create_contractor_from_expr2t_not(const expr2tc &expr)
{
  ibex::Ctc *contractor = nullptr;
  auto base_object = get_base_object(expr);

  if (is_not2t(expr))
  {
    auto not_ = to_not2t(base_object);
    contractor = create_contractor_from_expr2t(not_.value);
  }
  else if (is_constraint_operator_not(expr))
  {
    auto cons = create_constraint_from_expr2t_not(base_object);
    if (cons == nullptr)
      return nullptr;
    contractor = new ibex::CtcFwdBwd(*cons);
    vector_ctc.push_back(contractor);
  }
  else
  {
    switch (base_object->expr_id)
    {
    case expr2t::expr_ids::or_id:
    {
      auto logic_op = to_or2t(base_object);
      auto side1 = create_contractor_from_expr2t_not(logic_op.side_1);
      auto side2 = create_contractor_from_expr2t_not(logic_op.side_2);
      if (side1 != nullptr && side2 != nullptr)
      {
        contractor = new ibex::CtcCompo(*side1, *side2);
        vector_ctc.push_back(contractor);
      }
      break;
    }
    case expr2t::expr_ids::and_id:
    {
      auto logic_op = to_and2t(base_object);
      auto side1 = create_contractor_from_expr2t_not(logic_op.side_1);
      auto side2 = create_contractor_from_expr2t_not(logic_op.side_2);
      if (side1 != nullptr && side2 != nullptr)
      {
        contractor = new ibex::CtcUnion(*side1, *side2);
        vector_ctc.push_back(contractor);
      }
      break;
    }
    case expr2t::expr_ids::equality_id:
    {
      auto rel = to_equality2t(base_object);
      ibex::Function *f = create_function_from_expr2t(rel.side_1);
      ibex::Function *g = create_function_from_expr2t(rel.side_2);
      if (g == nullptr || f == nullptr)
        return nullptr;
      auto *side1 = new ibex::NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
      auto *side2 = new ibex::NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
      auto *c_side1 = new ibex::CtcFwdBwd(*side1);
      auto *c_side2 = new ibex::CtcFwdBwd(*side2);
      contractor = new ibex::CtcUnion(*c_side1, *c_side2);

      //for clean up
      vector_nc.push_back(side1);
      vector_nc.push_back(side2);
      vector_ctc.push_back(c_side1);
      vector_ctc.push_back(c_side2);
      vector_ctc.push_back(contractor);
      break;
    }
    case expr2t::expr_ids::typecast_id:
    {
      contractor =
        create_contractor_from_expr2t_not(get_base_object(base_object));
      break;
    }
    default:
      parse_error(base_object);
      break;
    }
  }
  return contractor;
}

bool expr_to_ibex_parser::is_constraint_operator(const expr2tc &expr)
{
  expr2tc e = get_base_object(expr);
  return (is_comp_expr(e) && !is_notequal2t(e)) || is_arith_expr(e);
}

bool expr_to_ibex_parser::is_constraint_operator_not(const expr2tc &expr)
{
  expr2tc e = get_base_object(expr);
  return (is_comp_expr(e) && !is_equality2t(e)) || is_arith_expr(e);
}

ibex::NumConstraint *
expr_to_ibex_parser::create_constraint_from_expr2t(const expr2tc &expr)
{
  ibex::NumConstraint *c = nullptr;
  auto base_object = get_base_object(expr);

  if (is_unsupported_operator_in_constraint(expr))
  {
    parse_error(expr);
    return nullptr;
  }

  if (is_not2t(expr))
  {
    c = create_constraint_from_expr2t_not(to_not2t(base_object).value);
    return c;
  }

  auto rel = &dynamic_cast<const relation_data &>(*get_base_object(expr));
  ibex::Function *f, *g;
  f = create_function_from_expr2t(rel->side_1);
  g = create_function_from_expr2t(rel->side_2);
  if (f == nullptr || g == nullptr)
    return nullptr;

  //change expression
  //replace
  switch (base_object->expr_id)
  {
  case expr2t::expr_ids::greaterthanequal_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) >= (*g)(*vars));
    break;
  case expr2t::expr_ids::greaterthan_id:
    //if int do
    //base_object->foreach_operand(op){};
    //c = new ibex::NumConstraint(*vars, (*f)(*vars) >= (*g)(*vars)+1);
    c = new ibex::NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
    break;
  case expr2t::expr_ids::lessthanequal_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) <= (*g)(*vars));
    break;
  case expr2t::expr_ids::lessthan_id:
    //c = new ibex::NumConstraint(*vars, (*f)(*vars) <= (*g)(*vars)-1);
    c = new ibex::NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
    break;
  case expr2t::expr_ids::equality_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) = (*g)(*vars));
    break;
  default:
    return nullptr;
  }
  //for clean up
  vector_nc.push_back(c);

  return c;
}

ibex::NumConstraint *
expr_to_ibex_parser::create_constraint_from_expr2t_not(const expr2tc &expr)
{
  ibex::NumConstraint *c;

  if (is_unsupported_operator_in_constraint_not(expr))
  {
    parse_error(expr);
    return nullptr;
  }
  if (is_not2t(expr))
  {
    c = create_constraint_from_expr2t(to_not2t(expr).value);
    return c;
  }

  auto rel = &dynamic_cast<const relation_data &>(*get_base_object(expr));

  ibex::Function *f, *g;
  f = create_function_from_expr2t(rel->side_1);
  g = create_function_from_expr2t(rel->side_2);
  if (f == nullptr || g == nullptr)
    return nullptr;
  switch (get_base_object(expr)->expr_id)
  {
  case expr2t::expr_ids::lessthan_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) >= (*g)(*vars));
    break;
  case expr2t::expr_ids::lessthanequal_id:
    //c = new ibex::NumConstraint(*vars, (*f)(*vars) >= (*g)(*vars)+1);
    c = new ibex::NumConstraint(*vars, (*f)(*vars) > (*g)(*vars));
    break;
  case expr2t::expr_ids::greaterthan_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) <= (*g)(*vars));
    break;
  case expr2t::expr_ids::greaterthanequal_id:
    //c = new ibex::NumConstraint(*vars, (*f)(*vars) <= (*g)(*vars)-1);
    c = new ibex::NumConstraint(*vars, (*f)(*vars) < (*g)(*vars));
    break;
  case expr2t::expr_ids::notequal_id:
    c = new ibex::NumConstraint(*vars, (*f)(*vars) = (*g)(*vars));
    break;
  default:
    return nullptr;
  }
  //for clean up
  vector_nc.push_back(c);

  return c;
}

bool expr_to_ibex_parser::is_unsupported_operator_in_constraint(
  const expr2tc &expr)
{
  expr2tc e = get_base_object(expr);
  return is_arith_expr(e) || is_constant_number(e) || is_symbol2t(e) ||
         is_notequal2t(e) || is_modulus2t(e) || is_or2t(e) || is_and2t(e);
}

bool expr_to_ibex_parser::is_unsupported_operator_in_constraint_not(
  const expr2tc &expr)
{
  expr2tc e = get_base_object(expr);
  return is_arith_expr(e) || is_constant_number(e) || is_symbol2t(e) ||
         is_equality2t(e) || is_modulus2t(e) || is_or2t(e) || is_and2t(e);
}

ibex::Function *expr_to_ibex_parser::create_function_from_expr2t(expr2tc expr)
{
  ibex::Function *f = nullptr;
  ibex::Function *g, *h;

  if (is_comp_expr(expr))
  {
    parse_error(expr);
    return nullptr;
  }
  switch (expr->expr_id)
  {
  case expr2t::expr_ids::add_id:
  case expr2t::expr_ids::sub_id:
  case expr2t::expr_ids::mul_id:
  case expr2t::expr_ids::div_id:
  {
    if (!is_arith_expr(expr))
      return nullptr;
    auto arith_op = &dynamic_cast<const arith_2ops &>(*get_base_object(expr));
    g = create_function_from_expr2t(arith_op->side_1);
    h = create_function_from_expr2t(arith_op->side_2);
    if (g == nullptr || h == nullptr)
      return nullptr;

    switch (arith_op->expr_id)
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
    if (h == nullptr)
      return nullptr;
    f = new ibex::Function(*vars, -(*h)(*vars));
    break;
  }
  case expr2t::expr_ids::symbol_id:
  {
    int index = create_variable_from_expr2t(expr);
    if (index == CspMap::NOT_FOUND)
    {
      log_error("MAX VAR SIZE REACHED");
      return nullptr;
    }
    else if (index == CspMap::IGNORE)
    {
      log_debug("contractor", "Expression contains ignored variable name");
      return nullptr;
    }
    else
      f = new ibex::Function(*vars, (*vars)[index]);
    break;
  }
  case expr2t::expr_ids::typecast_id:
    f = create_function_from_expr2t(to_typecast2t(expr).from);
    return f;
    break;
  case expr2t::expr_ids::constant_int_id:
  {
    const ibex::ExprConstant &c =
      ibex::ExprConstant::new_scalar(to_constant_int2t(expr).value.to_int64());
    f = new ibex::Function(*vars, c);
    break;
  }
  case expr2t::expr_ids::constant_floatbv_id:
  {
    const ibex::ExprConstant &c = ibex::ExprConstant::new_scalar(
      to_constant_floatbv2t(expr).value.to_double());
    f = new ibex::Function(*vars, c);
    break;
  }
  default:
    f = nullptr;
  }
  //for clean up
  vector_f.push_back(f);

  return f;
}

int expr_to_ibex_parser::create_variable_from_expr2t(expr2tc expr)
{
  if (is_symbol2t(expr))
  {
    std::string var_name = to_symbol2t(expr).get_symbol_name();
    if (
      var_name == "c:stdlib.c@__ESBMC_atexits" || var_name == "NULL" ||
      has_prefix(var_name, "c:@__ESBMC"))
      return map->IGNORE;
    int index = map->add_var(var_name, expr);
    return index;
  }
  return map->NOT_FOUND;
}

void expr_to_ibex_parser::parse_error(const expr2tc &expr)
{
  std::ostringstream oss;
  oss << get_expr_id(expr);
  oss << " Expression has unsupported operator, skipping this expression.\n";
  log_debug("contractor", "{}", oss.str());
}

void interval_analysis_ibex_contractor::maps_to_domains(
  int_mapt int_map,
  real_mapt real_map)
{
  auto t_0 = std::chrono::steady_clock::now();
  auto it = int_map.begin();
  while (it != int_map.end())
  {
    auto index = map.find(it->first.as_string());
    if (index == CspMap::NOT_FOUND)
    {
      it++;
      continue;
    }

    if (it->second.lower)
    {
      
      map.update_lb_interval(
        it->second.get_lower().to_int64(), it->first.as_string());
    }

    if (it->second.upper)
    {
      map.update_ub_interval(
        it->second.get_upper().to_int64(), it->first.as_string());
    }
    it++;
  }

  auto rit = real_map.begin();
  while (rit != real_map.end())
  {
    if (rit->second.lower)
    {
      map.update_lb_interval(
        (double)rit->second.get_lower(), rit->first.as_string());
    }
    if (rit->second.upper)
    {
      map.update_ub_interval(
        (double)rit->second.get_upper(), rit->first.as_string());
    }
    rit++;
  }
  cpy_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_0)
      .count();
}

void interval_analysis_ibex_contractor::apply_contractor()
{
  auto t_0 = std::chrono::steady_clock::now();
  auto X = map.create_interval_vector();
  auto c_out = contractor.get_outer();
  auto Y = X;
  int i = 50;

  c_out->contract(X);

  //Find a fixed point in 5 or fewer iterations
  while (Y != X && i >= 0)
  {
    Y = X;
    c_out->contract(X);
    i--;
  }

  //copy results to map_outer
  map_outer = CspMap(map);
  map_outer.update_intervals(X);
  apply_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_0)
      .count();
}

expr2tc interval_analysis_ibex_contractor::result_of_outer(expr2tc exp)
{
  expr2tc cond = gen_true_expr();
  // "x > = y "
  // "x >= y && x >= 0 && x <=10"
  // "true && x >= 0 && x <=10"

  if (map_outer.is_empty_set())
    return gen_false_expr();
  else
  {
    for (auto const &var : map_outer.var_map)
    {
      //testing with updating both bounds after reduction.
      expr2tc X = var.second.getSymbol();

      //if empty, skip
      if (var.second.getInterval().is_empty())
        continue;

      //if unbounded, skip
      if (
        isinf(var.second.getInterval().lb()) &&
        isinf(var.second.getInterval().ub()))
        continue;

      //if there is a lower bound,
      if (isnormal(var.second.getInterval().lb()))
      {
        //check if it overflows when cast back to integer
        BigInt r(0);
        if (is_signedbv_type(X->type))
        {
          r.setPower2(X->type->get_width() - 1);
          r = -r;
        } // if its unsigned then its just zero

        BigInt integerValue(0);

        if (var.second.getInterval().lb() < r.to_int64())
          integerValue = r;
        else
          integerValue = (long)ceil(var.second.getInterval().lb());

        auto lb = constant_int2tc(X->type, integerValue);
        auto cond1 = greaterthanequal2tc(X, lb);
        cond = and2tc(cond, cond1);
      }
      if (isnormal(var.second.getInterval().ub()))
      {
        BigInt r(0);
        r.setPower2(
          X->type->get_width() - (is_unsignedbv_type(X->type) ? 0 : 1));
        r = r - 1;

        BigInt integerValue(0);

        if (var.second.getInterval().ub() > r.to_uint64())
          integerValue = r;
        else
          integerValue = (long)floor(var.second.getInterval().ub());

        auto ub = constant_int2tc(X->type, integerValue);
        auto cond2 = lessthanequal2tc(X, ub);
        cond = and2tc(cond, cond2);
      }
    }
  }
  return cond;
}

void interval_analysis_ibex_contractor::dump()
{
  std::ostringstream oss;
  auto number_of_vars = map.size();
  auto x1 = map.create_interval_vector();
  auto x2 = map_outer.create_interval_vector();

  oss << "------------------------Contractor stats:\n";
  oss << "constraint :" << to_oss(contractor.get_outer()).str() << "\n";
  oss << "number of variables: " << number_of_vars << "\n";
  oss << "before : " << x1 << " diam: " << x1.diam() << std::endl;
  oss << "after : " << x2 << " diam: " << x2.diam() << std::endl;
  //oss << "Contractor parse time: " << parse_time << "\n";
  //oss << "Contractor maps_to_domains time: " << cpy_time << "\n";
  //oss << "Contractor modularize time: " << mod_time << "\n";
  //oss << "Contractor contract time: " << apply_time << "\n";
  oss << "------------------------";
  map.dump();
  map_outer.dump();
  log_status("{}\n\n", oss.str());
}

[[maybe_unused]] void interval_analysis_ibex_contractor::modularize_intervals()
{
  auto t_0 = std::chrono::steady_clock::now();
  std::ostringstream oss;
  for (auto &var : map.var_map)
  {
    if (!var.second.getInterval().is_unbounded())
      continue;

    auto v = to_symbol2t(var.second.getSymbol());
    auto i = var.second.getInterval();
    auto width = v.type->get_width();
    oss << v.get_symbol_name() << " width = " << width << "\n";
    log_status("{}", oss.str());

    double lb = i.lb(), ub = i.ub();

    if (is_floatbv_type(v.type))
    {
      if (isinf(i.lb()))
        lb = -std::numeric_limits<double>::max();
      if (isinf(i.ub()))
        ub = std::numeric_limits<double>::max();
    }
    else
    {
      if (isinf(i.lb()))
        lb = std::numeric_limits<long>::min();

      if (isinf(i.ub()))
        ub = std::numeric_limits<unsigned long>::max();
    }
    ibex::Interval new_interval(lb, ub);
    var.second.setInterval(new_interval);
  }
  mod_time =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - t_0)
      .count();
}
