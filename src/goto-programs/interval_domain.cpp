/// \file
/// Interval Domain

#include <goto-programs/interval_domain.h>
#include <langapi/language_util.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>

void interval_domaint::output(std::ostream &out) const
{
  if(bottom)
  {
    out << "BOTTOM\n";
    return;
  }

  for(const auto &interval : int_map)
  {
    if(interval.second.is_top())
      continue;
    if(interval.second.lower_set)
      out << interval.second.lower << " <= ";
    out << interval.first;
    if(interval.second.upper_set)
      out << " <= " << interval.second.upper;
    out << "\n";
  }
}

void interval_domaint::transform(
  goto_programt::const_targett from,
  goto_programt::const_targett to,
  ai_baset &,
  const namespacet &ns)
{
  (void)ns;

  const goto_programt::instructiont &instruction = *from;
  switch(instruction.type)
  {
  case DECL:
    havoc_rec(instruction.code);
    break;

  case ASSIGN:
    assign(instruction.code);
    break;

  case GOTO:
  {
    // Comparing iterators is safe as the target must be within the same list
    // of instructions because this is a GOTO.
    goto_programt::const_targett next = from;
    next++;
    if(from->targets.front() != next) // If equal then a skip
    {
      if(next == to)
      {
        expr2tc guard = instruction.guard;
        make_not(guard);
        assume(guard);        
      }
      else
        assume(instruction.guard);        
    }
  }
  break;

  case ASSUME:
    assume(instruction.guard);
    break;

  case FUNCTION_CALL:
  {
    const code_function_call2t &code_function_call =
      to_code_function_call2t(instruction.code);
    if(!is_nil_expr(code_function_call.ret))
      havoc_rec(code_function_call.ret);
  }
  break;

  default:;
  }
}

bool interval_domaint::join(const interval_domaint &b)
{
  if(b.bottom)
    return false;
  if(bottom)
  {
    *this = b;
    return true;
  }

  bool result = false;

  for(int_mapt::iterator it = int_map.begin(); it != int_map.end();) // no it++
  {
    // search for the variable that needs to be merged
    // containers have different size and variable order
    const int_mapt::const_iterator b_it = b.int_map.find(it->first);
    if(b_it == b.int_map.end())
    {
      it = int_map.erase(it);
      result = true;
    }
    else
    {
      integer_intervalt previous = it->second;
      it->second.join(b_it->second);
      if(it->second != previous)
        result = true;

      it++;
    }
  }

  return result;
}

void interval_domaint::assign(const expr2tc &expr)
{
  assert(is_code_assign2t(expr));
  auto const &c = to_code_assign2t(expr);
  havoc_rec(c.target);
  assume_rec(c.target, expr2t::equality_id, c.source);
}

void interval_domaint::havoc_rec(const expr2tc &expr)
{
  if(is_if2t(expr))
  {
    havoc_rec(to_if2t(expr).true_value);
    havoc_rec(to_if2t(expr).false_value);
  }
  else if(is_symbol2t(expr))
  {
    irep_idt identifier = to_symbol2t(expr).thename;

    if(is_bv_type(expr))
      int_map.erase(identifier);
  }
  else if(is_typecast2t(expr))
  {
    havoc_rec(to_typecast2t(expr).from);
  }
  else
    log_debug("[havoc_rec] Missing support: {}", *expr);
}

void interval_domaint::assume_rec(
  const expr2tc &lhs,
  expr2t::expr_ids id,
  const expr2tc &rhs)
{
  if(is_typecast2t(lhs))
    return assume_rec(to_typecast2t(lhs).from, id, rhs);

  if(is_typecast2t(rhs))
    return assume_rec(lhs, id, to_typecast2t(rhs).from);

  if(id == expr2t::equality_id)
  {
    assume_rec(lhs, expr2t::greaterthanequal_id, rhs);
    assume_rec(lhs, expr2t::lessthanequal_id, rhs);
    return;
  }

  if(id == expr2t::notequal_id)
    return; // won't do split

  if(id == expr2t::greaterthanequal_id)
    return assume_rec(rhs, expr2t::lessthanequal_id, lhs);

  if(id == expr2t::greaterthan_id)
    return assume_rec(rhs, expr2t::lessthan_id, lhs);

  // we now have lhs <  rhs or
  //             lhs <= rhs

  assert(id == expr2t::lessthan_id || id == expr2t::lessthanequal_id);

  if(is_symbol2t(lhs) && is_constant_number(rhs))
  {
    irep_idt lhs_identifier = to_symbol2t(lhs).thename;

    if(is_bv_type(lhs) && is_bv_type(rhs))
    {
      BigInt tmp = to_constant_int2t(rhs).value;
      if(id == expr2t::lessthan_id)
        --tmp;
      integer_intervalt &ii = int_map[lhs_identifier];
      ii.make_le_than(tmp);
      if(ii.is_bottom())
        make_bottom();
    }
  }
  else if(is_constant_number(lhs) && is_symbol2t(rhs))
  {
    irep_idt rhs_identifier = to_symbol2t(rhs).thename;

    if(is_bv_type(lhs) && is_bv_type(rhs))
    {
      BigInt tmp = to_constant_int2t(lhs).value;
      if(id == expr2t::lessthan_id)
        ++tmp;
      integer_intervalt &ii = int_map[rhs_identifier];
      ii.make_ge_than(tmp);
      if(ii.is_bottom())
        make_bottom();
    }
  }
  else if(is_symbol2t(lhs) && is_symbol2t(rhs))
  {   
    irep_idt lhs_identifier = to_symbol2t(lhs).thename;
    irep_idt rhs_identifier = to_symbol2t(rhs).thename;
    // This does not work for nondet!
    if(rhs_identifier.as_string().find("__VERIFIER_nondet"))
      return;
    if(is_bv_type(lhs) && is_bv_type(rhs))
    {
      integer_intervalt &lhs_i = int_map[lhs_identifier];
      integer_intervalt &rhs_i = int_map[rhs_identifier];      
      lhs_i.meet(rhs_i);
      rhs_i = lhs_i;
      if(rhs_i.is_bottom())
        make_bottom();
    }
  }
}

void interval_domaint::assume(const expr2tc &cond)
{
  expr2tc new_cond = cond;
  simplify(new_cond);
  assume_rec(new_cond, false);
}

void interval_domaint::assume_rec(const expr2tc &cond, bool negation)
{
  if(is_comp_expr(cond))
  {
    assert(cond->get_num_sub_exprs() == 2);

    if(negation) // !x<y  ---> x>=y
    {
      if(is_lessthan2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::greaterthanequal_id,
          *cond->get_sub_expr(1));
      else if(is_lessthanequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::greaterthan_id,
          *cond->get_sub_expr(1));
      else if(is_greaterthan2t(cond))
        assume_rec(
          *cond->get_sub_expr(0),
          expr2t::lessthanequal_id,
          *cond->get_sub_expr(1));
      else if(is_greaterthanequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::lessthan_id, *cond->get_sub_expr(1));
      else if(is_equality2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::notequal_id, *cond->get_sub_expr(1));
      else if(is_notequal2t(cond))
        assume_rec(
          *cond->get_sub_expr(0), expr2t::equality_id, *cond->get_sub_expr(1));
    }
    else
      assume_rec(*cond->get_sub_expr(0), cond->expr_id, *cond->get_sub_expr(1));
  }
  else if(is_not2t(cond))
  {
    assume_rec(to_not2t(cond).value, !negation);
  }
  else if(is_and2t(cond))
  {
    if(!negation)
      cond->foreach_operand([this](const expr2tc &e) { assume_rec(e, false); });
  }
  else if(is_or2t(cond))
  {
    if(negation)
      cond->foreach_operand([this](const expr2tc &e) { assume_rec(e, true); });
  }
  else
    log_debug("[assume_rec] Missing support: {}", *cond);
}

void interval_domaint::dump() const
{
  std::ostringstream oss;
  output(oss);
  log_debug("{}", oss.str());
}
expr2tc interval_domaint::make_expression(const expr2tc &symbol) const
{
  assert(is_symbol2t(symbol));

  symbol2t src = to_symbol2t(symbol);
  if(is_bv_type(symbol))
  {
    int_mapt::const_iterator i_it = int_map.find(src.thename);
    if(i_it == int_map.end())
      return gen_true_expr();

    const integer_intervalt &interval = i_it->second;
    if(interval.is_top())
      return gen_true_expr();

    if(interval.is_bottom())
      return gen_false_expr();

    std::vector<expr2tc> conjuncts;
    if(interval.singleton())
    {
      expr2tc value = from_integer(interval.upper, src.type);
      expr2tc new_expr = symbol;
      c_implicit_typecast_arithmetic(
        new_expr, value, *migrate_namespace_lookup);
      conjuncts.push_back(equality2tc(new_expr, value));
    }
    else
    {
      if(interval.upper_set)
      {
        expr2tc value = from_integer(interval.upper, src.type);
        expr2tc new_expr = symbol;
        c_implicit_typecast_arithmetic(
          new_expr, value, *migrate_namespace_lookup);
        conjuncts.push_back(lessthanequal2tc(new_expr, value));
      }

      if(interval.lower_set)
      {
        expr2tc value = from_integer(interval.lower, src.type);
        expr2tc new_expr = symbol;
        c_implicit_typecast_arithmetic(
          new_expr, value, *migrate_namespace_lookup);
        conjuncts.push_back(lessthanequal2tc(value, new_expr));
      }
    }

    return conjunction(conjuncts);
  }

  return gen_true_expr();
}

bool interval_domaint::ai_simplify(expr2tc &condition, const namespacet &ns)
  const
{
  (void)ns;

  bool unchanged = true;
  interval_domaint d(*this);

  // merge intervals to properly handle conjunction
  if(is_and2t(condition)) // May be directly representable
  {
    interval_domaint a;
    a.make_top();        // a is everything
    a.assume(condition); // Restrict a to an over-approximation
                         //  of when condition is true
    if(!a.join(d))       // If d (this) is included in a...
    {                    // Then the condition is always true
      unchanged = is_true(condition);
      condition = gen_true_expr();
    }
  }
  else if(is_symbol2t(condition))
  {
    // TODO: we have to handle symbol expression
  }
  else // Less likely to be representable
  {
    expr2tc not_condition = condition;
    make_not(not_condition);
    d.assume(not_condition); // Restrict to when condition is false
    if(d.is_bottom())        // If there there are none...
    {                        // Then the condition is always true
      unchanged = is_true(condition);
      condition = gen_true_expr();
    }
  }

  return unchanged;
}
