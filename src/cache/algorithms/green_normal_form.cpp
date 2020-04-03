//
// Created by rafaelsa on 13/03/2020.
//

#include "green_normal_form.h"

void green_normal_form::run_on_assert(
  symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;

  // First assert irep should begin with an implies
  assert(cond->expr_id == expr2t::expr_ids::implies_id);

  // LHS only holds the guard which is not useful. So we parse RHS
  std::shared_ptr<logic_2ops> implies;
  implies = std::dynamic_pointer_cast<logic_2ops>(cond);
  expr2tc &rhs(implies->side_2);
  process_expr(rhs);
}

bool green_normal_form::is_integer_expr(expr2tc &relation)
{
  if(relation->expr_id == expr2t::expr_ids::add_id)
  {
    // Process inner expression
  }

  if(relation->expr_id == expr2t::expr_ids::symbol_id)
  {
    return true;
  }

  if(relation->expr_id == expr2t::expr_ids::constant_int_id)
  {
    return true;
  }

  return false;
}


bool green_normal_form::is_operator_correct(expr2tc &relation)
{
  return true;
}


void green_normal_form::convert_to_normal_form(expr2tc &equality)
{
  
}

void green_normal_form::set_rightest_value_of_lhs_relation(expr2tc &equality, BigInt value)
{
  if(equality->expr_id == expr2t::expr_ids::add_id)
  {
    std::shared_ptr<arith_2ops> arith;
    arith = std::dynamic_pointer_cast<arith_2ops>(equality);
    set_rightest_value_of_lhs_relation(arith->side_2, value);
  }
  else if(equality->expr_id == expr2t::expr_ids::constant_int_id)
  {
    std::shared_ptr<constant_int2t> relation_rhs;
    relation_rhs = std::dynamic_pointer_cast<constant_int2t>(equality); 
    relation_rhs->value -= value;
  }

}

void green_normal_form::process_expr(expr2tc &rhs)
{
  if (rhs->expr_id != expr2t::expr_ids::equality_id)
  {
    return;
  }

  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(rhs);

  // Set RHS to 0
  BigInt old_rhs;
  if(relation->side_2->expr_id != expr2t::expr_ids::constant_int_id)
  {
    return;
  }

  std::shared_ptr<constant_int2t> relation_rhs;
  relation_rhs = std::dynamic_pointer_cast<constant_int2t>(relation->side_2); 
  old_rhs = relation_rhs->value;
  relation_rhs->value = 0;


}
