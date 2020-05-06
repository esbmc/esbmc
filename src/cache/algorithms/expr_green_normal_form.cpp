// Rafael Sá Menezes - 04/2020

#include <cache/algorithms/expr_green_normal_form.h>

#include <util/irep2_expr.h>
#include <array>

namespace
{
std::array<expr2t::expr_ids, 3> relations_ok = {
  expr2t::expr_ids::notequal_id,
  expr2t::expr_ids::equality_id,
  expr2t::expr_ids::lessthanequal_id};

std::array<expr2t::expr_ids, 3> relations_not_ok = {
  expr2t::expr_ids::lessthan_id,
  expr2t::expr_ids::greaterthan_id,
  expr2t::expr_ids::greaterthanequal_id};

void subtract_rhs_of_lhs(const expr2tc &expr)
{
  // TODO: Add an algorithm to simplify expressions
  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);

  auto type = relation->side_1->type;
  sub2tc subtraction(type, relation->side_1, relation->side_2);

  relation->side_1 = subtraction;

  BigInt zero(0);
  signedbv_type2tc i32type(32); // TODO: Not sure if this will cause problems
  auto zero_constant = constant_int2tc(i32type, zero);

  relation->side_2 = zero_constant;
}
} // namespace

expr2tc expr_green_normal_form::convert_inequality(expr2tc &inequality)
{
  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);

  BigInt one(1);
  BigInt neg_one(-1);
  auto i32type =
    signedbv_type2tc(32); // TODO: Not sure if this will cause problems
  auto one_constant = constant_int2tc(i32type, one);
  auto neg_one_constant = constant_int2tc(i32type, neg_one);

  switch(inequality->expr_id)
  {
  case expr2t::expr_ids::lessthan_id:
    // To convert from < to <= we simply subtract 1 from LHS
    {
      sub2tc subtraction(i32type, relation->side_1, one_constant);
      return lessthanequal2tc(subtraction, relation->side_2);
    }
  case expr2t::expr_ids::greaterthan_id:
    // To convert from > to <= we subtract 1 from LHS and them multiply it
    // to -1
    {
      sub2tc subtraction(i32type, relation->side_1, one_constant);
      mul2tc multiplication(i32type, neg_one_constant, subtraction);
      return lessthanequal2tc(multiplication, relation->side_2);
    }
  case expr2t::expr_ids::greaterthanequal_id:
    // To convert from >= to <= we only multiply for -1
    {
      mul2tc multiplication(i32type, neg_one_constant, relation->side_1);
      return lessthanequal2tc(multiplication, relation->side_2);
    }
    break;
  default:
    // Based on previous logic, this should only occur in the specified operators
    assert(0);
  }
}

void expr_green_normal_form::run()
{
  /**
   * 1 - LHS = LHS - RHS, RHS = 0
   * 2 - Check relation operator
   */

  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);
  // Only run in relations!
  if(!relation)
    return;

  // 1 - LHS = LHS - RHS, RHS = 0
  subtract_rhs_of_lhs(expr);

  // 2 - Check relation operator and convert supported relations
  bool is_conversion_needed =
    std::find(
      relations_not_ok.begin(), relations_not_ok.end(), this->expr->expr_id) !=
    relations_not_ok.end();
  if(is_conversion_needed)
    expr = this->convert_inequality(this->expr);

  bool expr_is_ok_relation =
    std::find(relations_ok.begin(), relations_ok.end(), this->expr->expr_id) !=
    relations_ok.end();

  if(!expr_is_ok_relation)
    return;
}
