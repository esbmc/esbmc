//
// Created by Rafael Sá Menezes on 07/04/20.
//

#include "ssa_flattener.h"
void ssa_flattener::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;

  // First assert irep should begin with an implies
  assert(cond->expr_id == expr2t::expr_ids::implies_id);

  std::shared_ptr<logic_2ops> implies;
  implies = std::dynamic_pointer_cast<logic_2ops>(cond);

  expr2tc &lhs(implies->side_1);
  std::shared_ptr<symbol_data> symbol;
  symbol = std::dynamic_pointer_cast<symbol_data>(lhs);
  std::string guard_name = symbol->get_symbol_name();

  expr2tc &rhs(implies->side_2);

  std::set<expr_hash> guard_items;

  // If the assertive contains inner guards then rhs will be an imply
  if(rhs->expr_id == expr2t::expr_ids::implies_id)
  {
    std::shared_ptr<logic_2ops> inner_implies;
    inner_implies = std::dynamic_pointer_cast<logic_2ops>(rhs);
    guard_items.insert(inner_implies->side_2->crc());
    parse_implication_guard(inner_implies->side_1, guard_items);
  }
  else
  {
    guard_items.insert(rhs->crc());
  }

  items[guard_name] = guard_items;

  //std::cout << "Checking if assertive guard was already proven...\n";
  if(gs.get(guard_items))
  {
    //step.ignore = true;
    constant_bool2tc false_value(true);
    step.cond = false_value;
  }
}

void ssa_flattener::run_on_assignment(symex_target_equationt::SSA_stept &step)
{
  if(step.hidden)
  {
    std::string guard_name;
    if(!is_guard(step.lhs, guard_name))
      return;

    expr2tc &rhs = step.rhs;
    auto relations = parse_guard(rhs);
    this->items[guard_name] = relations;
  }
}

void ssa_flattener::parse_implication_guard(
  const expr2tc &expr,
  std::set<expr_hash> &inner_items)
{
  if(expr->expr_id == expr2t::expr_ids::and_id)
  {
    std::shared_ptr<logic_2ops> and_expr;
    and_expr = std::dynamic_pointer_cast<logic_2ops>(expr);
    parse_implication_guard(and_expr->side_1, inner_items);
    parse_implication_guard(and_expr->side_2, inner_items);
  }

  std::string guard_name;
  if(is_guard(expr, guard_name))
  {
    for(const auto &item : items[guard_name])
    {
      inner_items.insert(item);
    }
  }
}

std::set<expr_hash> ssa_flattener::parse_guard(const expr2tc &expr)
{
  std::set<expr_hash> local_items;
  if(expr->expr_id != expr2t::expr_ids::equality_id)
    return local_items;

  std::shared_ptr<relation_data> relation;
  relation = std::dynamic_pointer_cast<relation_data>(expr);

  local_items.insert(relation->crc());
  return local_items;
}

bool ssa_flattener::is_guard(const expr2tc &expr, std::string &name)
{
  if(expr->expr_id != expr2t::expr_ids::symbol_id)
    return false;

  std::shared_ptr<symbol_data> symbol;
  symbol = std::dynamic_pointer_cast<symbol_data>(expr);
  std::string symbol_name = symbol->get_symbol_name();
  name = symbol_name;
  return symbol_name.find("goto_symex::guard") != std::string::npos;
}
