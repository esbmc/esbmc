/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <goto-symex/dynamic_allocation.h>
#include <goto-symex/goto_symex.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/irep2.h>
#include <util/std_expr.h>

void goto_symext::default_replace_dynamic_allocation(expr2tc &expr)
{

  expr->Foreach_operand([this] (expr2tc &e) {
    if (!is_nil_expr(e))
      default_replace_dynamic_allocation(e);
     }
   );

  if (is_valid_object2t(expr))
  {
    // replace with CPROVER_alloc[POINTER_OBJECT(...)]
    const valid_object2t &obj = to_valid_object2t(expr);

    pointer_object2tc obj_expr(pointer_type2(), obj.value);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(ns.lookup(valid_ptr_arr_name)), alloc_arr_2);

    index2tc index_expr(get_bool_type(), alloc_arr_2, obj_expr);
    expr = index_expr;
  }
  else if (is_invalid_pointer2t(expr))
  {
    const invalid_pointer2t &ptr = to_invalid_pointer2t(expr);

    pointer_object2tc obj_expr(pointer_type2(), ptr.ptr_obj);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(ns.lookup(valid_ptr_arr_name)), alloc_arr_2);

    index2tc index_expr(get_bool_type(), alloc_arr_2, obj_expr);
    not2tc notindex(index_expr);

    // XXXjmorse - currently we don't correctly track the fact that stack
    // objects change validity as the program progresses, and the solver is
    // free to guess that a stack ptr is invalid, as we never update
    // __ESBMC_alloc for stack ptrs.
    // So, add the precondition that invalid_ptr only ever applies to dynamic
    // objects.

    expr2tc sym_2;
    migrate_expr(symbol_expr(ns.lookup(dyn_info_arr_name)), sym_2);

    pointer_object2tc ptr_obj(pointer_type2(), ptr.ptr_obj);
    index2tc is_dyn(get_bool_type(), sym_2, ptr_obj);

    // Catch free pointers: don't allow anything to be pointer object 1, the
    // invalid pointer.
    type2tc ptr_type = type2tc(new pointer_type2t(type2tc(new empty_type2t())));
    symbol2tc invalid_object(ptr_type, "INVALID");
    equality2tc isinvalid(ptr.ptr_obj, invalid_object);
    not2tc notinvalid(isinvalid);

    and2tc is_not_bad_ptr(notindex, is_dyn);
    or2tc is_valid_ptr(is_not_bad_ptr, isinvalid);

    expr = is_valid_ptr;
  }
  if (is_deallocated_obj2t(expr))
  {
    // replace with CPROVER_alloc[POINTER_OBJECT(...)]
    const deallocated_obj2t &obj = to_deallocated_obj2t(expr);

    pointer_object2tc obj_expr(pointer_type2(), obj.value);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(ns.lookup(deallocd_arr_name)), alloc_arr_2);

    index2tc index_expr(get_bool_type(), alloc_arr_2, obj_expr);
    expr = index_expr;
  }
  else if (is_dynamic_size2t(expr))
  {
    // replace with CPROVER_alloc_size[POINTER_OBJECT(...)]
    //nec: ex37.c
    const dynamic_size2t &size = to_dynamic_size2t(expr);

    pointer_object2tc obj_expr(pointer_type2(), size.value);

    expr2tc alloc_arr_2;
    migrate_expr(symbol_expr(ns.lookup(alloc_size_arr_name)), alloc_arr_2);

    index2tc index_expr(uint_type2(), alloc_arr_2, obj_expr);
    expr = index_expr;
  }
}
