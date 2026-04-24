#include <goto-symex/goto_symex.h>
#include <string>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/migrate.h>
#include <util/std_types.h>

void goto_symext::symex_va_arg(
  const expr2tc &lhs,
  const sideeffect2t &code [[maybe_unused]],
  const guardt &guard)
{
  std::string base =
    id2string(cur_state->top().function_identifier) + "::va_arg";

  irep_idt id = base + std::to_string(cur_state->top().va_index++);

  expr2tc va_rhs;

  const symbolt *s = new_context.find_symbol(id);
  if (s != nullptr)
  {
    type2tc symbol_type = migrate_type(s->type);

    va_rhs = symbol2tc(symbol_type, s->id);
    cur_state->top().level1.get_ident_name(va_rhs);

    va_rhs = typecast2tc(lhs->type, va_rhs);
  }
  else
  {
    va_rhs = gen_zero(lhs->type);
  }

  symex_assign(code_assign2tc(lhs, va_rhs), true, guard);
}
