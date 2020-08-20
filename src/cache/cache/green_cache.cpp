// Rafael Sá Menezes - 04/2020

#include <cache/cache/green_cache.h>
#include <cache/algorithms/algorithms.h>
#include <cache/algorithms/algorithms_utils.h>

void green_cache::canonize_expr(expr2tc &expr)
{
  if(apply_reordering)
  {
    expr_variable_reordering reordering(expr);
    reordering.run();
  }

  if(apply_normalization)
  {
    expr_green_normal_form gnf(expr);
    gnf.run();
  }
  // TODO: add variable renaming
}

void green_cache::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;

  // First assert irep should begin with an implies
  if(cond->expr_id != expr2t::expr_ids::implies_id)
  {
    cond->dump();
    return;
  }

  std::shared_ptr<logic_2ops> implies;
  implies = std::dynamic_pointer_cast<logic_2ops>(cond);

  expr2tc &lhs(implies->side_1);
  std::shared_ptr<symbol_data> symbol;
  symbol = std::dynamic_pointer_cast<symbol_data>(lhs);
  std::string guard_name = symbol->get_symbol_name();

  expr2tc &rhs(implies->side_2);

  crc_expr guard_items;

  // If the assertive contains inner guards then rhs will be an imply
  if(rhs->expr_id == expr2t::expr_ids::implies_id)
  {
    /*
     * If an ASSERTION contains an inner implies this means that this is in the
     * format:
     *
     * GUARD0 && GUARD1 && ... && GUARDN => ¬inner_expr
     *
     * LHS does not need to be canonized and can contain || besides &&
     * TODO: Add support to ||
     */
    std::shared_ptr<logic_2ops> inner_implies;
    inner_implies = std::dynamic_pointer_cast<logic_2ops>(rhs);

    // Canonize inner_expr
    canonize_expr(inner_implies->side_2);
    auto inner_expr_hash = convert_expr_to_hash(inner_implies->side_2);
    guard_items.insert(inner_expr_hash);
    parse_implication_guard(inner_implies->side_1, guard_items);
  }
  else
  {
    // Simple guard
    canonize_expr(rhs);
    auto expr_hash = convert_expr_to_hash(rhs);
    guard_items.insert(expr_hash);
  }

  items.insert({guard_name, guard_items});

  load_unsat_container();
  if(unsat_container.check(guard_items))
  {
    /**
     * An assertion is in the format of
     *
     * a. assertion_guard -> !expr
     * b. assertion_guard -> guards && ... && guard -> !expr.
     *
     * For 'a' the is the trivial case, if expr is known to be false
     * then we mark it's negation as true.
     *
     * In 'b', using basic logic equivalence:
     *
     * A -> B <-> ¬A OR B
     *
     * Where A would be the set of guards and B the !expr. Since A if false
     * it's negation is true, so we can simplify to:
     *
     * assertion_guard -> 1 (true)
     *
     */
    constant_bool2tc false_value(true);
    step.cond = false_value;
  }
}

void green_cache::parse_implication_guard(
  const expr2tc &expr,
  crc_expr &inner_items)
{
  if(expr->expr_id == expr2t::expr_ids::and_id)
  {
    // This should be executed to each guard.
    std::shared_ptr<logic_2ops> and_expr;
    // TODO: Add support to ||
    and_expr = std::dynamic_pointer_cast<logic_2ops>(expr);
    parse_implication_guard(and_expr->side_1, inner_items);
    parse_implication_guard(and_expr->side_2, inner_items);
  }

  // Recursively get to the last element which should be a guard
  std::string guard_name;
  if(expr_algorithm_util::is_guard(expr, guard_name))
  {
    for(const auto &item : items[guard_name])
    {
      inner_items.insert(item);
    }
  }
  else
  {
    // Note: Not sure if the guard always implies in a group of guards
    //assert(0);
  }
}

void green_cache::run_on_assignment(symex_target_equationt::SSA_stept &step)
{
  // Guards are hidden
  if(step.hidden)
  {
    std::string guard_name;
    if(!expr_algorithm_util::is_guard(step.lhs, guard_name))
      return;

    // Canonize rhs
    canonize_expr(step.rhs);

    expr2tc &rhs = step.rhs;
    crc_expr relations = parse_guard(rhs);
    // Adds it to the dictionary
    items.insert({guard_name, relations});
  }
}

crc_hash green_cache::convert_expr_to_hash(const expr2tc &expr)
{
  return expr->crc();
}

crc_expr green_cache::parse_guard(const expr2tc &expr)
{
  //assert(expr->expr_id == expr2t::expr_ids::equality_id);
  crc_expr local_items;
  // TODO: support other relations (<=, !=)
  //if(expr->expr_id != expr2t::expr_ids::equality_id)
  //return local_items;

  // Just get the hash from the expr and add it
  local_items.insert(convert_expr_to_hash(expr));
  return local_items;
}

void green_cache::load_unsat_container()
{
  // Load default unsat cache
  std::string filename("unsat_database");
  text_file_crc_set_storage storage(filename);
  //unsat_container.set(storage.load());
}

void green_cache::mark_ssa_as_unsat()
{
  // Load a default file
  std::string filename("unsat_database");
  text_file_crc_set_storage storage(filename);
  for(const auto &[key, value] : items)
  {
    // Adds all items from the SSA to the unsat container
    unsat_container.add(value);
  }
  // Stores it in the cache
  //storage.store(unsat_container);
}