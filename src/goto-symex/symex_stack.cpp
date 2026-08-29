#include <goto-symex/goto_symex.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>

void goto_symex_statet::framet::grow_stack_frame(const expr2tc &expr)
{
  stack_frame_total += (type_byte_size(expr->type) * 8);
}

void goto_symex_statet::framet::decrease_stack_frame_size(const expr2tc &expr)
{
  const code_dead2t &decl_code = to_code_dead2t(expr);

  // Obtain the width of the dead expression and decrease it from the
  // total number of bits for a given stack frame.
  stack_frame_total -= (type_byte_size(decl_code.type) * 8);
}

BigInt goto_symex_statet::total_stack_size() const
{
  BigInt total(0);
  for (const framet &frame : call_stack)
    // Hidden frames are ESBMC's own operational models (memset, the pthread
    // wrappers, ...). Their locals are a modelling artefact, not storage the
    // target program spends, so counting them would make the limit
    // impossible to calibrate against a real stack budget.
    if (!frame.hidden)
      total += frame.stack_frame_total;
  return total;
}

static expr2tc stack_size_claim(const BigInt &size, unsigned long limit)
{
  return lessthanequal2tc(
    constant_int2tc(get_uint64_type(), size),
    constant_int2tc(get_uint64_type(), BigInt(limit)));
}

void goto_symext::check_stack_size(
  const expr2tc &expr,
  const std::string &subject)
{
  cur_state->top().grow_stack_frame(expr);

  if (stack_limit > 0)
    claim(
      stack_size_claim(cur_state->top().stack_frame_total, stack_limit),
      "Stack limit property was violated" + subject);

  if (total_stack_limit > 0)
    claim(
      stack_size_claim(cur_state->total_stack_size(), total_stack_limit),
      "Total stack limit property was violated" + subject);
}
