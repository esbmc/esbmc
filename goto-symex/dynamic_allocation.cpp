/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <cprover_prefix.h>
#include <expr_util.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "dynamic_allocation.h"

/*******************************************************************\

Function: replace_dynamic_allocation

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void replace_dynamic_allocation(
  const namespacet &ns,
  exprt &expr)
{
  Forall_operands(it, expr)
    replace_dynamic_allocation(ns, *it);

  if(expr.id()=="valid_object")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()==typet::t_pointer);

    // replace with CPROVER_alloc[POINTER_OBJECT(...)]

    exprt object_expr("pointer_object", uint_type());
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "alloc"));

    exprt index_expr(exprt::index, typet(typet::t_bool));
    index_expr.move_to_operands(alloc_array, object_expr);

    expr.swap(index_expr);
  }
  if (expr.id()=="invalid-pointer")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()=="pointer");

    exprt theptr = expr.op0();

    exprt object_expr("pointer_object", uint_type());
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "alloc"));

    exprt index_expr("index", typet("bool"));
    index_expr.move_to_operands(alloc_array, object_expr);

    exprt notindex("not", bool_typet());
    notindex.move_to_operands(index_expr);

    // XXXjmorse - currently we don't correctly track the fact that stack
    // objects change validity as the program progresses, and the solver is
    // free to guess that a stack ptr is invalid, as we never update
    // __ESBMC_alloc for stack ptrs.
    // So, add the precondition that invalid_ptr only ever applies to dynamic
    // objects.

    exprt is_dyn("is_dynamic_object", bool_typet());
    is_dyn.move_to_operands(theptr);
    exprt is_valid_ptr("and", bool_typet());
    is_valid_ptr.move_to_operands(notindex, is_dyn);

    expr.swap(is_valid_ptr);
  }
  if(expr.id()=="deallocated_object")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()==typet::t_pointer);

    // replace with CPROVER_alloc[POINTER_OBJECT(...)]

    exprt object_expr("pointer_object", uint_type());
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "deallocated"));

    exprt index_expr("memory-leak", typet(typet::t_bool));
    index_expr.move_to_operands(alloc_array, object_expr);

    expr.swap(index_expr);
  }
  else if(expr.id()=="dynamic_size")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()==typet::t_pointer);

    // replace with CPROVER_alloc_size[POINTER_OBJECT(...)]
    //nec: ex37.c
    exprt object_expr("pointer_object", int_type()/*uint_type()*/);
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "alloc_size"));

    exprt index_expr(exprt::index, ns.follow(alloc_array.type()).subtype());
    index_expr.move_to_operands(alloc_array, object_expr);

    expr.swap(index_expr);
  }
  else if(expr.id()=="pointer_object_has_type")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()==typet::t_pointer);

  }
}
