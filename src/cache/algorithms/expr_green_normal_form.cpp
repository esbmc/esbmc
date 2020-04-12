// Rafael Sá Menezes - 04/2020

#include <util/irep2_expr.h>
#include "expr_green_normal_form.test.h"

namespace
{
constexpr std::array<expr2t::expr_ids, 3> relations_ok = {
  expr2t::expr_ids::notequal_id,
  expr2t::expr_ids::equality_id,
  expr2t::expr_ids::lessthanequal_id};

constexpr std::array<expr2t::expr_ids, 3> relations_not_ok = {
  expr2t::expr_ids::lessthan_id,
  expr2t::expr_ids::greaterthan_id,
  expr2t::expr_ids::greaterthanequal_id};
} // namespace

void expr_green_normal_form::convert_inequality(expr2tc &inequality)
{
  // TODO: Add support to conversions
  return;
}

void expr_green_normal_form::run()
{
  /**
   * 1 - Check current relation and apply conversions
   * 2 - Set RHS as 0
   * 3 - Apply difference from RHS to LHS
   */

  // 1 - Check current relation and apply conversions
  // Check if a conversion is needed
  bool is_conversion_needed =
    std::find(
      relations_not_ok.begin(), relations_not_ok.end(), this->expr->expr_id) !=
    relations_not_ok.end();
  if(is_conversion_needed)
    this->convert_inequality(this->expr);

  bool expr_is_ok_relation =
    std::find(relations_ok.begin(), relations_ok.end(), this->expr->expr_id) !=
    relations_ok.end();

  if(!expr_is_ok_relation)
    return;

  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);

  // 2 - Set RHS as 0

  if(relation->side_2->expr_id != expr2t::expr_ids::constant_int_id)
  {
    // TODO: Add support to expressions in RHS
    return;
  }

  std::shared_ptr<constant_int2t> relation_rhs;
  relation_rhs = std::dynamic_pointer_cast<constant_int2t>(relation->side_2);
  BigInt old_rhs = relation_rhs->value;
  relation_rhs->value = 0;

  // 3 - Apply difference from RHS to LHS
  set_rightest_value_of_lhs_relation(relation->side_1, old_rhs);
}

void expr_green_normal_form::set_rightest_value_of_lhs_relation(
  expr2tc &equality,
  BigInt value)
{
  // TODO: Add support to other binary operations
  // TODO: Add support to expressions without k

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