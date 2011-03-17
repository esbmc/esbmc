/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <cprover_prefix.h>
#include <expr_util.h>

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
    assert(expr.op0().type().id()=="pointer");

    // replace with CPROVER_alloc[POINTER_OBJECT(...)]

    exprt object_expr("pointer_object", uint_type());
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "alloc"));

    exprt index_expr("index", typet("bool"));
    index_expr.move_to_operands(alloc_array, object_expr);

    expr.swap(index_expr);
  }
  if(expr.id()=="deallocated_object")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()=="pointer");

    // replace with CPROVER_alloc[POINTER_OBJECT(...)]

    exprt object_expr("pointer_object", uint_type());
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "alloc"));

    exprt index_expr("memory-leak", typet("bool"));
    index_expr.move_to_operands(alloc_array, object_expr);

    expr.swap(index_expr);
  }
  else if(expr.id()=="dynamic_size")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()=="pointer");

    // replace with CPROVER_alloc_size[POINTER_OBJECT(...)]
    //nec: ex37.c
    exprt object_expr("pointer_object", int_type()/*uint_type()*/);
    object_expr.move_to_operands(expr.op0());

    exprt alloc_array=symbol_expr(ns.lookup(CPROVER_PREFIX "alloc_size"));

    exprt index_expr("index", ns.follow(alloc_array.type()).subtype());
    index_expr.move_to_operands(alloc_array, object_expr);

    expr.swap(index_expr);
  }
  else if(expr.id()=="pointer_object_has_type")
  {
    assert(expr.operands().size()==1);
    assert(expr.op0().type().id()=="pointer");

  }
}
