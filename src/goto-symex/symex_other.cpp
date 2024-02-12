#include <cassert>
#include <goto-symex/goto_symex.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/pretty.h>

void goto_symext::symex_other(const expr2tc code)
{
  expr2tc code2 = code;
  if (is_code_expression2t(code2))
  {
    // Represents an expression that gets evaluated, but does not have any
    // other effect on execution, i.e. doesn't contain a call or assignment.
    // This can, however, cause the program to fail if it dereferences an
    // invalid pointer. Therefore, dereference it.
    const code_expression2t &expr = to_code_expression2t(code2);
    expr2tc operand = expr.operand;
    dereference(operand, dereferencet::READ);
  }
  else if (is_code_cpp_del_array2t(code2) || is_code_cpp_delete2t(code2))
  {
    replace_dynamic_allocation(code2);
    replace_nondet(code2);

    symex_cpp_delete(code2);
  }
  else if (is_code_free2t(code2))
  {
    symex_free(code2);
  }
  else if (is_code_printf2t(code2))
  {
    replace_dynamic_allocation(code2);
    replace_nondet(code2);
    dereference(code2, dereferencet::READ);
    symex_printf(expr2tc(), code2);
  }
  else if (is_code_asm2t(code2))
  {
    // Assembly statement -> do nothing.
    return;
  }
  else
    throw "goto_symext: unexpected statement: " + get_expr_id(code2);
}

void goto_symext::symex_decl(const expr2tc code)
{
  assert(is_code_decl2t(code));

  expr2tc code2 = code;
  replace_dynamic_allocation(code2);
  replace_nondet(code2);
  dereference(code2, dereferencet::READ);

  // check whether the stack limit check has been activated.
  if (stack_limit > 0)
  {
    // extract the actual variable name.
    const std::string pretty_name =
      get_pretty_name(to_code_decl2t(code).value.as_string());

    // check whether the stack size has been reached.
    claim(
      (cur_state->top().process_stack_size(code2, stack_limit)),
      "Stack limit property was violated when declaring " + pretty_name);
  }

  const code_decl2t &decl_code = to_code_decl2t(code2);

  // just do the L2 renaming to preseve locality
  const irep_idt &identifier = decl_code.value;

  // Generate dummy symbol as a vehicle for renaming.
  expr2tc l1_sym = symbol2tc(get_empty_type(), identifier);

  // increase the frame if we have seen this declaration before
  statet::framet &frame = cur_state->top();
  do
  {
    unsigned &index = cur_state->variable_instance_nums[identifier];
    frame.level1.rename(l1_sym, ++index);
    to_symbol2t(l1_sym).level1_num = index;
  } while (frame.declaration_history.find(renaming::level2t::name_record(
             to_symbol2t(l1_sym))) != frame.declaration_history.end());

  // Rename it to the new name
  cur_state->top().level1.get_ident_name(l1_sym);

  // And record it
  renaming::level2t::name_record l(to_symbol2t(l1_sym));
  frame.declaration_history.insert(l);
  frame.local_variables.insert(l);

  // seen it before?
  // it should get a fresh value
  if (cur_state->level2.current_number(l1_sym) != 0)
  {
    // Dummy assignment - blank constant value isn't considered for const
    // propagation, variable number will be bumped to result in a new free
    // variable. Invalidates l1_symbol reference?
    cur_state->level2.make_assignment(l1_sym, expr2tc(), expr2tc());
  }
}

void goto_symext::symex_dead(const expr2tc code)
{
  assert(is_code_dead2t(code));

  expr2tc code2 = code;
  replace_dynamic_allocation(code2);
  replace_nondet(code2);
  dereference(code2, dereferencet::INTERNAL);

  // check whether the stack limit check has been activated.
  if (stack_limit > 0)
    cur_state->top().decrease_stack_frame_size(code2);

  const code_dead2t &dead_code = to_code_dead2t(code2);

  // just do the L2 renaming to preseve locality
  const irep_idt &identifier = dead_code.value;

  // Generate dummy symbol as a vehicle for renaming.
  expr2tc l1_sym = symbol2tc(dead_code.type, identifier);

  // Rename it to level 1
  cur_state->top().level1.get_ident_name(l1_sym);

  // Call free on alloca'd objects
  if (identifier.as_string().find("return_value$_alloca") != std::string::npos)
    symex_free(code_free2tc(l1_sym));

  // Erase from level 1 propagation
  cur_state->value_set.erase(to_symbol2t(l1_sym).get_symbol_name());

  // Erase from local_variables map
  cur_state->top().local_variables.erase(
    renaming::level2t::name_record(to_symbol2t(l1_sym)));
}
