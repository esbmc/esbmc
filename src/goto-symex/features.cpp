#include "irep2/irep2_type.h"
#include <goto-symex/features.h>

bool ssa_features::run(symex_target_equationt::SSA_stepst &steps)
{

  features.clear();

  for (const auto &step : steps)
  {
    const expr2tc expressions[2] = {step.cond, step.guard};
    for (const expr2tc &e : expressions)
      check(e);
  }

  print_result();

  return false;
}

void ssa_features::check(const expr2tc &e)
{
  if (!e)
    return;

  // TODO: We could add a cache
  if (is_array_type(e->type))
    features.insert(SSA_FEATURES::ARRAY);
    

  if (is_struct_type(e->type))
    features.insert(SSA_FEATURES::STRUCTS);


  switch (e->expr_id)
  {
  case expr2t::constant_fixedbv_id:
  case expr2t::constant_floatbv_id:
    features.insert(SSA_FEATURES::NON_INTEGER_NUMERAL);
    break;
    
  case expr2t::expr_ids::mul_id:
  case expr2t::expr_ids::div_id:
  case expr2t::modulus_id:
    if (!(is_constant_expr(dynamic_cast<const arith_2ops &>(*e).side_1) ||
          is_constant_expr(dynamic_cast<const arith_2ops &>(*e).side_2)))
      features.insert(SSA_FEATURES::NON_LINEAR);
    
    break;

  case expr2t::shl_id:
  case expr2t::ashr_id:
  case expr2t::lshr_id:
  case expr2t::bitor_id:
  case expr2t::bitand_id:
  case expr2t::bitxor_id:
  case expr2t::bitnand_id:
  case expr2t::bitnor_id:
  case expr2t::bitnxor_id:
    // TODO: shifts are linear if shifted by a constant size
    features.insert(SSA_FEATURES::NON_LINEAR);
    features.insert(SSA_FEATURES::BITWISE_OPERATIONS);    
    break;

  case expr2t::overflow_cast_id:
  case expr2t::overflow_id:
  case expr2t::overflow_neg_id:
    features.insert(SSA_FEATURES::OVERFLOW_ASSERTIONS);    
    break;

  default:
    break;
  }

  e->foreach_operand([this](const expr2tc &next) {
    check(next);
});
}


void ssa_features::print_result() const
{
  if (features.count(SSA_FEATURES::NON_LINEAR))
    log_status("SSA: Contains NON LINEAR");
  if (features.count(SSA_FEATURES::NON_INTEGER_NUMERAL))
    log_status("SSA: Contains NON INTEGER NUMERAL");
  if (features.count(SSA_FEATURES::BITWISE_OPERATIONS))
    log_status("SSA: Contains BITWISE OPERATIONS");
  if (features.count(SSA_FEATURES::OVERFLOW_ASSERTIONS))
    log_status("SSA: Contains OVERFLOW ASSERTIONS");
  if (features.count(SSA_FEATURES::ARRAY))
    log_status("SSA: Contains ARRAY");
  if (features.count(SSA_FEATURES::STRUCTS))
    log_status("SSA: Contains STRUCTS");
}
