#include <goto-programs/lift_call_expressions.h>
#include <irep2/irep2_utils.h>
#include <util/irep/migrate.h>
#include <util/symtab/symbol_generator.h>

namespace
{
/// The first expression-context call below `expr`. Migration spells these
/// `sideeffect2t`/function_call, so a statement-level `code_function_call2t`
/// is never one of them and needs no special case.
expr2tc find_nested_call(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return expr2tc();

  if (
    is_sideeffect2t(expr) &&
    to_sideeffect2t(expr).kind == sideeffect2t::allockind::function_call)
    return expr;

  expr2tc found;
  expr->foreach_operand([&found](const expr2tc &op) {
    if (is_nil_expr(found))
      found = find_nested_call(op);
  });
  return found;
}

/// Substitute `from` by `to` wherever it occurs below `expr`.
void substitute(expr2tc &expr, const expr2tc &from, const expr2tc &to)
{
  if (is_nil_expr(expr))
    return;

  if (expr == from)
  {
    expr = to;
    return;
  }

  expr->Foreach_operand(
    [&from, &to](expr2tc &op) { substitute(op, from, to); });
}
} // namespace

void lift_call_expressions(contextt &context, goto_functionst &goto_functions)
{
  symbol_generator gen("__ESBMC_lifted_call$");
  bool lifted = false;

  for (auto &[name, fn] : goto_functions.function_map)
  {
    (void)name;
    if (!fn.body_available)
      continue;

    Forall_goto_program_instructions (it, fn.body)
    {
      // One instruction can hide more than one, and substituting the first
      // can expose the next, so drain each instruction before moving on.
      for (;;)
      {
        expr2tc nested = find_nested_call(it->code);
        if (is_nil_expr(nested))
          nested = find_nested_call(it->guard);
        if (is_nil_expr(nested))
          break;

        const sideeffect2t &call = to_sideeffect2t(nested);
        const type2tc &ret_type = call.type;

        symbolt &tmp =
          gen.new_symbol(context, migrate_type_back(ret_type), "lifted");
        expr2tc tmp_expr = symbol2tc(ret_type, tmp.id);

        substitute(it->code, nested, tmp_expr);
        substitute(it->guard, nested, tmp_expr);

        goto_programt::targett call_inst = fn.body.insert(it);
        call_inst->type = FUNCTION_CALL;
        call_inst->location = it->location;
        call_inst->code =
          code_function_call2tc(tmp_expr, call.operand, call.arguments);
        lifted = true;

        // The callee is an ESBMC intrinsic that symex answers by name, so it
        // needs no body -- but goto_inline resolves every call through
        // function_map first and errors out on a name that is not there. A
        // native goto-binary compiles no C, so nothing else declares it.
        const irep_idt callee = to_symbol2t(call.operand).thename;
        if (!goto_functions.function_map.count(callee))
        {
          goto_functiont &decl = goto_functions.function_map[callee];
          decl.type = call.operand->type;
          decl.body_available = false;
        }
        if (!context.find_symbol(callee))
        {
          symbolt fsym;
          fsym.mode = "C";
          fsym.name = callee;
          fsym.id = callee;
          fsym.set_type(migrate_type_back(call.operand->type));
          fsym.is_extern = true;
          context.add(fsym);
        }
      }
    }
  }

  if (lifted)
    goto_functions.update();
}
