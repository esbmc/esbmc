#include "dirty.h"

void dirtyt::build(const goto_functiont &goto_function)
{
  for(const auto &i : goto_function.body.instructions)
  {
    if(i.is_other())
      search_other(i);
    else if(i.code)
      i.code->foreach_operand([this](const expr2tc &e) { find_dirty(e); });
  }
}

void dirtyt::search_other(const goto_programt::instructiont &instruction)
{
  assert(instruction.is_other() && "instruction type must be OTHER");
  if(is_code_expression2t(instruction.code))
    instruction.code->foreach_operand(
      [this](const expr2tc &e) { find_dirty(e); });
}

void dirtyt::find_dirty(const expr2tc &expr)
{
  if(!expr)
    return;
  if(is_address_of2t(expr))
  {
    find_dirty_address_of(to_address_of2t(expr).ptr_obj);
    return;
  }

  expr->foreach_operand([this](const expr2tc &e) { find_dirty(e); });
}

void dirtyt::find_dirty_address_of(const expr2tc &expr)
{
  if(is_symbol2t(expr))
  {
    log_debug(
      "[dirty] inserting symbol {}", to_symbol2t(expr).get_symbol_name());
    dirty.insert(to_symbol2t(expr).get_symbol_name());
  }
  else if(is_member2t(expr))
    find_dirty_address_of(to_member2t(expr).source_value);
  else if(is_index2t(expr))
  {
    find_dirty_address_of(to_index2t(expr).source_value);
    find_dirty(to_index2t(expr).index);
  }
  else if(is_dereference2t(expr))
    find_dirty(to_dereference2t(expr).value);
  else if(is_if2t(expr))
  {
    find_dirty_address_of(to_if2t(expr).true_value);
    find_dirty_address_of(to_if2t(expr).false_value);
    find_dirty(to_if2t(expr).cond);
  }
}

void dirtyt::output(std::ostream &out) const
{
  log_debug("[dirty] printing Dirty variables");
  for(const auto &d : dirty)
    out << d << '\n';
}

void incremental_dirtyt::populate_dirty_for_function(
  const std::string &id,
  const goto_functiont &function)
{
  auto insert_result = dirty_processed_functions.insert(id);
  if(insert_result.second)
    dirty.add_function(function);
}