/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <assert.h>

#include <cprover_prefix.h>
#include <expr_util.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "goto_symex.h"
#include "dynamic_allocation.h"

/*******************************************************************\

Function: default_replace_dynamic_allocation

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::default_replace_dynamic_allocation(expr2tc &expr)
{

  std::vector<expr2tc *> operands;
  expr.get()->list_operands(operands);
  for (std::vector<expr2tc *>::const_iterator it = operands.begin();
       it != operands.end(); it++)
    default_replace_dynamic_allocation(**it);

  if (is_valid_object2t(expr))
  {
    // replace with CPROVER_alloc[POINTER_OBJECT(...)]
    const valid_object2t &obj = to_valid_object2t(expr);

    expr2tc obj_expr(new pointer_object2t(uint_type2(), obj.value));

    exprt alloc_array=symbol_expr(ns.lookup(valid_ptr_arr_name));
    expr2tc alloc_arr_2;
    migrate_expr(alloc_array, alloc_arr_2);

    expr2tc index_expr = expr2tc(new index2t(type_pool.get_bool(),
                                              alloc_arr_2, obj_expr));
    expr = index_expr;
  }
  else if (is_invalid_pointer2t(expr))
  {
    const invalid_pointer2t &ptr = to_invalid_pointer2t(expr);

    expr2tc obj_expr(new pointer_object2t(uint_type2(), ptr.ptr_obj));

    exprt alloc_array=symbol_expr(ns.lookup(valid_ptr_arr_name));
    expr2tc alloc_arr_2;
    migrate_expr(alloc_array, alloc_arr_2);

    expr2tc index_expr = expr2tc(new index2t(type_pool.get_bool(),
                                             alloc_arr_2, obj_expr));
    expr2tc notindex = expr2tc(new not2t(index_expr));

    // XXXjmorse - currently we don't correctly track the fact that stack
    // objects change validity as the program progresses, and the solver is
    // free to guess that a stack ptr is invalid, as we never update
    // __ESBMC_alloc for stack ptrs.
    // So, add the precondition that invalid_ptr only ever applies to dynamic
    // objects.

    exprt sym = symbol_expr(ns.lookup(dyn_info_arr_name));
    expr2tc sym_2;
    migrate_expr(sym, sym_2);

    expr2tc ptr_obj = expr2tc(new pointer_object2t(int_type2(), ptr.ptr_obj));
    expr2tc is_dyn = expr2tc(new index2t(type_pool.get_bool(), sym_2, ptr_obj));

    // Catch free pointers: don't allow anything to be pointer object 1, the
    // invalid pointer.
    type2tc ptr_type = type2tc(new pointer_type2t(type2tc(new empty_type2t())));
    expr2tc invalid_object = expr2tc(new symbol2t(ptr_type, "INVALID"));
    expr2tc isinvalid = expr2tc(new equality2t(ptr.ptr_obj, invalid_object));
    expr2tc notinvalid = expr2tc(new not2t(isinvalid));

    expr2tc is_not_bad_ptr = expr2tc(new and2t(notindex, is_dyn));
    expr2tc is_valid_ptr = expr2tc(new or2t(is_not_bad_ptr, isinvalid));

    expr = is_valid_ptr;
  }
  if (is_deallocated_obj2t(expr))
  {
    // replace with CPROVER_alloc[POINTER_OBJECT(...)]
    const deallocated_obj2t &obj = to_deallocated_obj2t(expr);

    expr2tc obj_expr = expr2tc(new pointer_object2t(uint_type2(), obj.value));

    exprt alloc_array=symbol_expr(ns.lookup(deallocd_arr_name));
    expr2tc alloc_arr_2;
    migrate_expr(alloc_array, alloc_arr_2);

    expr2tc index_expr = expr2tc(new index2t(type_pool.get_bool(),
                                             alloc_arr_2, obj_expr));
    expr = index_expr;
  }
  else if (is_dynamic_size2t(expr))
  {
    // replace with CPROVER_alloc_size[POINTER_OBJECT(...)]
    //nec: ex37.c
    const dynamic_size2t &size = to_dynamic_size2t(expr);

    expr2tc obj_expr = expr2tc(new pointer_object2t(int_type2(), size.value));

    exprt alloc_array=symbol_expr(ns.lookup(alloc_size_arr_name));
    expr2tc alloc_arr_2;
    migrate_expr(alloc_array, alloc_arr_2);

    expr2tc index_expr = expr2tc(new index2t(uint_type2(), alloc_arr_2,
                                             obj_expr));
    expr = index_expr;
  }
}
