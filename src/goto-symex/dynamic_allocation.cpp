#include <cassert>
#include <goto-symex/dynamic_allocation.h>
#include <goto-symex/goto_symex.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <irep2/irep2.h>
#include <util/std_expr.h>

static inline void convert_capability_member(
  expr2tc &expr,
  const expr2tc &value,
  const irep_idt &field_name,
  const namespacet &ns)
{
  // Construct POINTER_CAPABILITY(...) from original value
  expr2tc cap_expr = pointer_capability2tc(ptraddr_type2(), value);

  expr2tc capability_arr;
  migrate_expr(
    symbol_expr(*ns.lookup("c:@__ESBMC_cheri_info")), capability_arr);

  // Get cheri_info[cap_expr]
  expr2tc index_expr = index2tc(
    to_array_type(capability_arr->type).subtype, capability_arr, cap_expr);

  // Access .base or .top member
  expr = member2tc(size_type2(), index_expr, field_name);
}

void goto_symext::default_replace_dynamic_allocation(expr2tc &expr)
{
  expr->Foreach_operand([this](expr2tc &e) {
    if (!is_nil_expr(e))
      default_replace_dynamic_allocation(e);
  });

  if (is_valid_object2t(expr))
  {
    /* alloc */
    // replace with CPROVER_alloc[POINTER_OBJECT(...)]
    const valid_object2t &obj = to_valid_object2t(expr);

    expr2tc obj_expr = pointer_object2tc(pointer_type2(), obj.value);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(*ns.lookup(valid_ptr_arr_name)), alloc_arr_2);

    expr2tc index_expr = index2tc(get_bool_type(), alloc_arr_2, obj_expr);
    expr = index_expr;
  }
  else if (is_invalid_pointer2t(expr))
  {
    /* (!valid /\ dynamic) \/ invalid */
    const invalid_pointer2t &ptr = to_invalid_pointer2t(expr);

    expr2tc obj_expr = pointer_object2tc(pointer_type2(), ptr.ptr_obj);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(*ns.lookup(valid_ptr_arr_name)), alloc_arr_2);

    expr2tc index_expr = index2tc(get_bool_type(), alloc_arr_2, obj_expr);
    expr2tc notindex = not2tc(index_expr);

    // XXXjmorse - currently we don't correctly track the fact that stack
    // objects change validity as the program progresses, and the solver is
    // free to guess that a stack ptr is invalid, as we never update
    // __ESBMC_alloc for stack ptrs.
    // So, add the precondition that invalid_ptr only ever applies to dynamic
    // objects.

    expr2tc sym_2;
    migrate_expr(symbol_expr(*ns.lookup(dyn_info_arr_name)), sym_2);

    expr2tc ptr_obj = pointer_object2tc(pointer_type2(), ptr.ptr_obj);
    expr2tc is_dyn = index2tc(get_bool_type(), sym_2, ptr_obj);

    // Catch free pointers: don't allow anything to be pointer object 1, the
    // invalid pointer.
    type2tc ptr_type = pointer_type2tc(get_empty_type());
    expr2tc invalid_object = symbol2tc(ptr_type, "INVALID");
    expr2tc isinvalid = equality2tc(ptr.ptr_obj, invalid_object);

    expr2tc is_not_bad_ptr = and2tc(notindex, is_dyn);
    expr2tc is_valid_ptr = or2tc(is_not_bad_ptr, isinvalid);

    expr = is_valid_ptr;
  }
  else if (is_deallocated_obj2t(expr))
  {
    /* !alloc */
    // replace with CPROVER_alloc[POINTER_OBJECT(...)]
    const deallocated_obj2t &obj = to_deallocated_obj2t(expr);

    expr2tc obj_expr = pointer_object2tc(pointer_type2(), obj.value);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(*ns.lookup(valid_ptr_arr_name)), alloc_arr_2);

    if (is_symbol2t(obj.value))
      expr = index2tc(get_bool_type(), alloc_arr_2, obj_expr);
    else
    {
      expr2tc index_expr = index2tc(get_bool_type(), alloc_arr_2, obj_expr);
      expr = not2tc(index_expr);
    }
  }
  else if (is_dynamic_size2t(expr))
  {
    // replace with CPROVER_alloc_size[POINTER_OBJECT(...)]
    //nec: ex37.c
    const dynamic_size2t &size = to_dynamic_size2t(expr);

    expr2tc obj_expr = pointer_object2tc(pointer_type2(), size.value);

    expr2tc alloc_arr_2;
    const symbolt* alloc_size_symbol = ns.lookup(alloc_size_arr_name);
    assert(alloc_size_symbol);
    migrate_expr(symbol_expr(*alloc_size_symbol), alloc_arr_2);

    expr2tc index_expr = index2tc(size_type2(), alloc_arr_2, obj_expr);
    expr = index_expr;
  }
  else if (is_capability_base2t(expr))
  {
    // replace with cheri_info[POINTER_CAPABILITY(...)].base
    const capability_base2t &size = to_capability_base2t(expr);

    convert_capability_member(expr, size.value, irep_idt("base"), ns);
  }
  else if (is_capability_top2t(expr))
  {
    // replace with cheri_info[POINTER_CAPABILITY(...)].top
    const capability_top2t &size = to_capability_top2t(expr);

    convert_capability_member(expr, size.value, irep_idt("top"), ns);
  }
}
