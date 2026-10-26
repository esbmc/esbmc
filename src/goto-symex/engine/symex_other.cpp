#include <cassert>
#include <goto-symex/engine/goto_symex.h>
#include <goto-symex/scheduler/reachability_tree.h>
#include <util/expr/expr_util.h>
#include <irep2/irep2.h>
#include <irep2/irep2_utils.h>
#include <util/symtab/pretty.h>

void goto_symext::symex_other(const expr2tc &code)
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

/// Numbers, or fixed-size aggregates of them. A pointer is left out: a
/// reused one would carry every iteration's targets into its value set. So
/// is a variable-length array, which may get another size each time.
static bool reusable_type(const type2tc &type)
{
  if (is_number_type(type))
    return true;
  if (is_array_type(type))
  {
    const array_type2t &arr = to_array_type(type);
    return !arr.size_is_infinite && is_constant_int2t(arr.array_size) &&
           reusable_type(arr.subtype);
  }
  if (is_struct_type(type) || is_union_type(type))
  {
    for (const type2tc &member : struct_union_members(type))
      if (!reusable_type(member))
        return false;
    return true;
  }
  return false;
}

void goto_symext::symex_decl(const expr2tc &code)
{
  assert(is_code_decl2t(code));

  expr2tc code2 = code;
  replace_dynamic_allocation(code2);
  replace_nondet(code2);
  dereference(code2, dereferencet::READ);

  // check whether either stack limit check has been activated.
  if (stack_checks_enabled())
    check_stack_size(
      code2,
      " when declaring " +
        get_pretty_name(to_code_decl2t(code).value.as_string()));

  const code_decl2t &decl_code = to_code_decl2t(code2);

  const irep_idt &identifier = decl_code.value;

  // Generate dummy symbol as a vehicle for renaming.
  expr2tc l1_sym = symbol2tc(get_empty_type(), identifier);

  // Keep the instance the frame gave this local on entry; a re-declaration
  // gets a fresh level-1 instance to preserve locality, unless its address
  // is never taken (array decay included, which ESBMC lowers to an explicit
  // address_of). No pointer can tell two instances of that apart, reaching a
  // declaration again re-initialises the same object (C11 6.2.4p6), and one
  // instance per loop iteration makes every later merge diff the stale ones
  // (#9150).
  statet::framet &frame = cur_state->top();
  frame.level1.get_ident_name(l1_sym);
  const renaming::level2t::name_record rec(to_symbol2t(l1_sym));
  const bool redeclared = frame.declaration_history.count(rec) != 0;
  const bool reused = redeclared && reusable_type(decl_code.type) &&
                      !art1->is_address_taken_local(identifier);
  if (reused)
    cur_state->value_set.erase_object(
      to_symbol2t(l1_sym).get_symbol_name(), decl_code.type);
  if (
    to_symbol2t(l1_sym).rlevel != symbol_renaming_level::level1 ||
    (redeclared && !reused))
  {
    unsigned &index = cur_state->variable_instance_nums[identifier];
    frame.level1.rename(l1_sym, ++index);
    frame.level1.get_ident_name(l1_sym);
  }

  // And record it
  renaming::level2t::name_record l(to_symbol2t(l1_sym));
  frame.declaration_history.insert(l);
  frame.local_variables.insert(l);

  // seen it before?
  // it should get a fresh value, even a reused instance its previous
  // lifetime never assigned (a call's result whose call was cut off)
  if (reused || cur_state->level2.current_number(l1_sym) != 0)
  {
    // Dummy assignment - blank constant value isn't considered for const
    // propagation, variable number will be bumped to result in a new free
    // variable. Invalidates l1_symbol reference?
    cur_state->level2.make_assignment(l1_sym, expr2tc(), expr2tc());
  }
}

void goto_symext::symex_dead(const expr2tc &code)
{
  assert(is_code_dead2t(code));

  expr2tc code2 = code;
  replace_dynamic_allocation(code2);
  replace_nondet(code2);
  dereference(code2, dereferencet::INTERNAL);

  const code_dead2t &dead_code = to_code_dead2t(code2);

  // drop the variable's L1 identity from value-set tracking and the
  // live-locals map
  const irep_idt &identifier = dead_code.value;

  // Generate dummy symbol as a vehicle for renaming.
  expr2tc l1_sym = symbol2tc(dead_code.type, identifier);

  // Rename it to level 1
  cur_state->top().level1.get_ident_name(l1_sym);
  const renaming::level2t::name_record rec(to_symbol2t(l1_sym));

  // A DEAD can run where its local is not live: on a short-circuit path that
  // never declared a loop condition's temporary, or twice on one path. Only
  // a live local still occupies stack.
  const bool live = cur_state->top().local_variables.erase(rec);
  if (live && stack_checks_enabled())
    cur_state->top().decrease_stack_frame_size(code2);

  cur_state->value_set.erase_object(
    to_symbol2t(l1_sym).get_symbol_name(), dead_code.type);
}
