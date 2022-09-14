#include <util/cache.h>
#include <util/message.h>
#include <utility>
#include <util/crypto_hash.h>

void assertion_cache::run_on_assert(symex_target_equationt::SSA_stept &step)
{
  auto [it, ins] = db.emplace(std::make_pair(step.cond, step.guard));
  if(!ins)
  {
    log_debug("Cache hits: {}", ++hits);
    step.cond = constant_bool2tc(trivial_value);
  }
  else
  {
    log_debug("Cache missed");
    if(match_subexpressions && step.cond->expr_id == expr2t::implies_id)
    {
      /* Assertions are in the form of step.guard => (exec_guard => expr)
       * meaning that we can cache the 'expr' as long as we don't touch 'exec_guard'
       */
      auto logic_op = std::dynamic_pointer_cast<logic_2ops>(step.cond);
      // First, add current expr into DB
      db.emplace(std::make_pair(logic_op->side_2, step.guard));
      // Now, check whether any of expr operands
      if(try_matching_sub_expression(logic_op->side_2, step.guard))
      {
        /* We changed the assertion, let's simplify and cache
         * the new condition */
        log_debug("Simplified a sub expression!");
        simplify(step.cond);
        auto [itt, inss] = db.emplace(std::make_pair(step.cond, step.guard));
        if(!inss)
        {
          log_debug("Cache hits: {}", ++hits);
          step.cond = constant_bool2tc(trivial_value);
        }
      };
    }
  }
}

bool assertion_cache::run(symex_target_equationt::SSA_stepst &eq)
{
  fine_timet algorithm_start = current_time();
  for(auto &step : eq)
    run_on_step(step);
  fine_timet algorithm_stop = current_time();
  log_status(
    "Caching time: {}s (removed {} assignments)",
    time2string(algorithm_stop - algorithm_start),
    hits);
  return true;
}

bool assertion_cache::try_matching_sub_expression(
  expr2tc &e,
  const expr2tc &guards)
{
  /* This will try matching any side of a logical
   * operator, e.g., and2t, or2t, implies2t
   */
  auto logic_op = std::dynamic_pointer_cast<logic_2ops>(e);
  bool matched = false;
  if(logic_op)
  {
    if(is_in_cache(logic_op->side_1, guards))
    {
      matched = true;
      log_debug("Sub-Cache hits: {}", ++sub_hits);
      logic_op->side_1 = constant_bool2tc(trivial_value);
    }
    else
      try_matching_sub_expression(logic_op->side_1, guards);

    if(is_in_cache(logic_op->side_2, guards))
    {
      matched = true;
      log_debug("Sub-Cache hits: {}", ++sub_hits);
      logic_op->side_2 = constant_bool2tc(trivial_value);
    }
    else
      try_matching_sub_expression(logic_op->side_2, guards);
  }
  return matched;
}
