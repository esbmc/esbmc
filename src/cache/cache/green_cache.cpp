// Rafael Sá Menezes - 04/2020

#include <cache/cache/green_cache.h>
#include <cache/algorithms/algorithms.h>
#include <cache/algorithms/algorithms_utils.h>

void green_cache::canonize_expr(expr2tc &expr)
{
  if(apply_reordering)
  {
    expr_variable_reordering reordering(expr);
    //reordering.run();
  }

  if(apply_normalization)
  {
    expr_green_normal_form gnf(expr);
    //gnf.run();
  }
}

void green_cache::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  expr2tc &cond = step.cond;

  // First assert irep should begin with an implies
  if(cond->expr_id != expr2t::expr_ids::implies_id)
  {
    cond->dump();
    // TODO: Fix this, a condition may be !guard
    return;
  }

  std::shared_ptr<logic_2ops> implies;
  implies = std::dynamic_pointer_cast<logic_2ops>(cond);

  expr2tc &lhs(implies->side_1);
  std::shared_ptr<symbol_data> symbol;
  symbol = std::dynamic_pointer_cast<symbol_data>(lhs);
  std::string guard_name = symbol->get_symbol_name();
  assertions.emplace(guard_name);
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
     * LHS does not need to be canonized and can contain ¬ and || besides &&
     */

    // TODO: Fix this
    std::cout << "Found an implication\n";
    return;
    std::shared_ptr<logic_2ops> inner_implies;
    inner_implies = std::dynamic_pointer_cast<logic_2ops>(rhs);

    // Canonize inner_expr
    canonize_expr(inner_implies->side_2);

    // This is needed
    auto not_p = not2tc(inner_implies->side_2);
    auto inner_expr_hash = convert_expr_to_hash(not_p);
    std::cout << "Inner hash " << inner_expr_hash << "\n";
    guard_items.insert(inner_expr_hash);
    parse_implication_guard(
      inner_implies->side_1, guard_items, inner_expr_hash);
  }
  else
  {
    /* This an assertion of the format
     * guard => ¬g
     */
    if(rhs->expr_id == expr2t::expr_ids::symbol_id)
    {
      std::shared_ptr<symbol_data> rhs_symbol;
      rhs_symbol = std::dynamic_pointer_cast<symbol_data>(rhs);
      std::string rhs_name = rhs_symbol->get_symbol_name();
      if(rhs_name.find("@F@assert") != std::string::npos)
      {
        return;
      }
    }
    // Simple guard
    canonize_expr(rhs);
    bool initialized = false;
    if(rhs->expr_id == expr2t::expr_ids::not_id)
    {
      std::string guard_name;
      if(expr_algorithm_util::is_guard(rhs, guard_name, true))
      {
        if(items.find(guard_name) == items.end())
        {
          std::cerr << "Guard " << guard_name << " wasn't parsed\n";
          abort();
        }
        for(const auto &item : items[guard_name])
        {
          guard_items.insert(item);
        }
        initialized = true;
      }
    }
    if(!initialized)
    {
      auto expr_hash = convert_expr_to_hash(rhs);
      guard_items.insert(expr_hash);
    }
  }

  //items.insert({guard_name, guard_items});
  this->to_add_container.emplace(guard_items);
  load_unsat_container();
  // Check the full assertion
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
  crc_expr &inner_items,
  crc_hash parent_property)
{
  if(expr->expr_id == expr2t::expr_ids::and_id)
  {
    // This should be executed to each guard.
    std::shared_ptr<logic_2ops> and_expr;
    and_expr = std::dynamic_pointer_cast<logic_2ops>(expr);
    crc_expr side_1;
    crc_expr side_2;
    parse_implication_guard(and_expr->side_1, side_1, parent_property);
    parse_implication_guard(and_expr->side_2, side_2, parent_property);

    // Check wether one of the sides is UNSAT
    load_unsat_container();
    if(unsat_container.check(side_1) || unsat_container.check(side_2))
    {
      // TODO: abort();
      constant_bool2tc false_value(false);
      and_expr->side_2 = false_value;
    }
    for(auto i : side_1)
      inner_items.insert(i);
    for(auto i : side_2)
      inner_items.insert(i);
    side_1.insert(parent_property);
    side_2.insert(parent_property);

    return;
  }

  else if(expr->expr_id == expr2t::expr_ids::or_id)
  {
    std::shared_ptr<logic_2ops> or_expr;
    or_expr = std::dynamic_pointer_cast<logic_2ops>(expr);
    crc_expr side_1;
    crc_expr side_2;
    parse_implication_guard(or_expr->side_1, side_1, parent_property);
    parse_implication_guard(or_expr->side_2, side_2, parent_property);

    load_unsat_container();
    if(unsat_container.check(side_1))
    {
      std::cout << "Hit OR simplification\n";
      // TODO: abort();
    }

    if(unsat_container.check(side_2))
    {
      std::cout << "Hit OR simplification\n";
      // TODO: abort();
    }

    inner_items.insert(expr->crc());
    return;
  }

  else if(expr->expr_id == expr2t::expr_ids::not_id)
  {
    std::cout << "Not expression\n";
    std::string guard_name;
    if(expr_algorithm_util::is_guard(expr, guard_name, true)) {
      if(unsat_container.check(items[guard_name])) {
        std::cout << "hit not cache\n";
        // TODO: abort();
        inner_items.insert(expr->crc());
      } else {
        inner_items.insert(expr->crc());
      }
    } else inner_items.insert(expr->crc());
    return;
  }

  else if(expr->expr_id == expr2t::expr_ids::constant_bool_id)
  {
    // TODO: abort();
    return;
  }
  // Recursively get to the last element which should be a guard
  std::string guard_name;
  if(expr_algorithm_util::is_guard(expr, guard_name))
  {
    if(items.find(guard_name) == items.end())
    {
      std::cerr << "Guard " << guard_name << " wasn't parsed\n";
      abort();
    }
    for(const auto &item : items[guard_name])
    {
      inner_items.insert(item);
    }
  }
  else
  {
    std::cerr << "This type of expression is not supported\n";
    expr->dump();
    abort();
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
    std::cout << "Got guard " << guard_name << ": " << rhs->crc() << "\n";
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
  //--enable-caching file.c --incremental-bmc --max-k-step 5.set(storage.load());
}

void green_cache::mark_ssa_as_unsat()
{
  std::cout << "Saving cache\n";
  for(const auto i : this->to_add_container)
  {
    std::cout << "ITEM\n";
      unsat_container.add(i);

  }

  std::cout << "Total size " << unsat_container.get_size() << "\n";
  // Stores it in the cache
  //storage.store(unsat_container);
}