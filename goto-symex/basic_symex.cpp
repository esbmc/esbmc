/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <base_type.h>
#include <simplify_expr.h>
#include <i2string.h>
#include <cprover_prefix.h>
#include <expr_util.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "basic_symex.h"
#include "dynamic_allocation.h"
#include "execution_state.h"

unsigned basic_symext::nondet_count=0;
unsigned basic_symext::dynamic_counter=0;

/*******************************************************************\

Function: basic_symext::assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::assignment(
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs)
{
  statet & state = ex_state.get_active_state();
  exprt original_lhs=lhs;
  state.get_original_name(original_lhs);

  exprt new_lhs=lhs;
  //replace_dynamic_allocation(state, rhs);
  //replace_nondet(rhs);

  state.assignment(new_lhs, rhs, ns, constant_propagation, ex_state, ex_state.node_id);

  target->assignment(
    state.guard,
    new_lhs, original_lhs,
    rhs,
    state.source,
    symex_targett::STATE);
}

/*******************************************************************\

Function: basic_symext::do_simplify

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::do_simplify(exprt &expr)
{
  if(options.get_bool_option("simplify"))
  {
    base_type(expr, ns);
    simplify(expr);
  }
}

/*******************************************************************\

Function: basic_symext::symex

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex(statet &state, execution_statet &ex_state, const codet &code,unsigned node_id)
{
  const irep_idt &statement=code.get("statement");

  if(statement=="block")
    symex_block(state, ex_state, code, node_id);
  else if(statement=="assign")
    symex_assign(state, ex_state, code, node_id);
  else if(statement=="decl")
  {
    // behaves like non-deterministic assignment
    if(code.operands().size()!=1)
      throw "decl expected to have one operand";

    exprt rhs("nondet_symbol", code.op0().type());
    rhs.set("identifier", "symex::nondet"+i2string(nondet_count++));
    rhs.location()=code.location();

    exprt new_lhs(code.op0());
    read(new_lhs);

    guardt guard; // NOT the state guard!
    symex_assign_rec(state, ex_state, new_lhs, rhs, guard,node_id);
  }
  else if(statement=="expression")
  {
    // ignore
  }
  else if(statement=="cpp_delete" ||
          statement=="cpp_delete[]")
    symex_cpp_delete(state, code);
  else if(statement=="free")
  {
    // like skip
  }
  else if(statement=="nondet")
  {
    // like skip
  }
  else if(statement=="printf")
    symex_printf(state, static_cast<const exprt &>(get_nil_irep()), code,node_id);
  else
  {
    std::cerr << code.pretty() << std::endl;
    throw "unexpected statement: "+id2string(statement);
  }
}

/*******************************************************************\

Function: basic_symext::symex_block

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_block(statet &state, execution_statet &ex_state, const codet &code,unsigned node_id)
{
  forall_operands(it, code)
    symex(state, ex_state, to_code(*it),node_id);
}

/*******************************************************************\

Function: basic_symext::symex_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign(statet &state, execution_statet &ex_state, const codet &code,unsigned node_id)
{
  if(code.operands().size()!=2)
    throw "assignment expects two operands";

  exprt lhs=code.op0();
  exprt rhs=code.op1();

  //replace_dynamic_allocation(state, lhs);
  //replace_dynamic_allocation(state, rhs);

  replace_nondet(lhs);
  replace_nondet(rhs);

  if(rhs.id()=="sideeffect")
  {
    const side_effect_exprt &side_effect_expr=to_side_effect_expr(rhs);
    const irep_idt &statement=side_effect_expr.get_statement();

    if(statement=="function_call")
    {
      assert(side_effect_expr.operands().size()!=0);

      if(side_effect_expr.op0().id()!="symbol")
        throw "symex_assign: expected symbol as function";

      const irep_idt &identifier=
        to_symbol_expr(side_effect_expr.op0()).get_identifier();

      throw "symex_assign: unexpected function call: "+id2string(identifier);
    }
    else if(statement=="cpp_new" ||
            statement=="cpp_new[]")
      symex_cpp_new(state, lhs, side_effect_expr, ex_state, node_id);
    else if(statement=="malloc")
      symex_malloc(state, lhs, side_effect_expr, ex_state, node_id);
    else if(statement=="printf")
      symex_printf(state, lhs, side_effect_expr,node_id);
    else
    {
      throw "symex_assign: unexpected sideeffect: "+id2string(statement);
    }
  }
  else
  {
    guardt guard; // NOT the state guard!
    symex_assign_rec(state, ex_state, lhs, rhs, guard,node_id);
  }
}

/*******************************************************************\

Function: basic_symext::read

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::read(exprt &expr)
{
}

/*******************************************************************\

Function: basic_symext::symex_assign_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_rec(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  if(lhs.id()=="symbol")
    symex_assign_symbol(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()=="index" || lhs.id()=="memory-leak")
    symex_assign_array(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()=="member")
    symex_assign_member(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()=="if")
    symex_assign_if(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()=="typecast")
    symex_assign_typecast(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()=="string-constant" ||
          lhs.id()=="NULL-object" ||
          lhs.id()=="zero_string")
  {
    // ignore
  }
  else if(lhs.id()=="byte_extract_little_endian" ||
          lhs.id()=="byte_extract_big_endian")
    symex_assign_byte_extract(state, ex_state, lhs, rhs, guard,node_id);
  else
    throw "assignment to "+lhs.id_string()+" not handled";
}

/*******************************************************************\

Function: basic_symext::symex_assign_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_symbol(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
  unsigned node_id)
{
  // put assignment guard in rhs
  if(!guard.empty())
  {
    exprt new_rhs("if", rhs.type());
    new_rhs.operands().resize(3);
    new_rhs.op0()=guard.as_expr();
    new_rhs.op1().swap(rhs);
    new_rhs.op2()=lhs;
    new_rhs.swap(rhs);
  }
  exprt original_lhs=lhs;
  state.get_original_name(original_lhs);
  state.rename(rhs, ns, node_id);
  do_simplify(rhs);

  exprt new_lhs=lhs;

  state.assignment(new_lhs, rhs, ns, constant_propagation, ex_state, node_id);

  guardt tmp_guard(state.guard);
  tmp_guard.append(guard);

  // do the assignment
  target->assignment(
    tmp_guard,
    new_lhs, original_lhs,
    rhs,
    state.source,
    symex_targett::STATE);

}

/*******************************************************************\

Function: basic_symext::symex_assign_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_typecast(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  // these may come from dereferencing on the lhs

  assert(lhs.operands().size()==1);

  exprt rhs_typecasted(rhs);

  rhs_typecasted.make_typecast(lhs.op0().type());

  symex_assign_rec(state, ex_state, lhs.op0(), rhs_typecasted, guard,node_id);
}

/*******************************************************************\

Function: basic_symext::symex_assign_array

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_array(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  // lhs must be index operand
  // that takes two operands: the first must be an array
  // the second is the index

  if(lhs.operands().size()!=2)
    throw "index must have two operands";

  const exprt &lhs_array=lhs.op0();
  const exprt &lhs_index=lhs.op1();
  const typet &lhs_type=ns.follow(lhs_array.type());

  if(lhs_type.id()!="array")
    throw "index must take array type operand";

  // turn
  //   a[i]=e
  // into
  //   a'==a WITH [i:=e]

  exprt new_rhs("with", lhs_type);

  new_rhs.reserve_operands(3);
  new_rhs.copy_to_operands(lhs_array);
  new_rhs.copy_to_operands(lhs_index);
  new_rhs.move_to_operands(rhs);

  symex_assign_rec(state, ex_state, lhs_array, new_rhs, guard,node_id);
}

/*******************************************************************\

Function: basic_symext::symex_assign_member

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_member(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  // symbolic execution of a struct member assignment

  // lhs must be member operand
  // that takes one operands, which must be a structure

  if(lhs.operands().size()!=1)
    throw "member must have one operand";

  exprt lhs_struct=lhs.op0();
  typet struct_type=ns.follow(lhs_struct.type());

  if(struct_type.id()!="struct" &&
     struct_type.id()!="union")
    throw "member must take struct/union type operand but got "
          +struct_type.pretty();

  const irep_idt &component_name=lhs.get("component_name");

  // typecasts involved? C++ does that for inheritance.
  if(lhs_struct.id()=="typecast")
  {
    assert(lhs_struct.operands().size()==1);

    if(lhs_struct.op0().id()=="NULL-object")
    {
      // ignore
    }
    else
    {
      // remove the type cast, we assume that the member is there
      exprt tmp(lhs_struct.op0());
      struct_type=ns.follow(tmp.type());
      assert(struct_type.id()=="struct" || struct_type.id()=="union");
      lhs_struct=tmp;
    }
  }

  // turn
  //   a.c=e
  // into
  //   a'==a WITH [c:=e]

  exprt new_rhs("with", struct_type);

  new_rhs.reserve_operands(3);
  new_rhs.copy_to_operands(lhs_struct);
  new_rhs.copy_to_operands(exprt("member_name"));
  new_rhs.move_to_operands(rhs);

  new_rhs.op1().set("component_name", component_name);

  symex_assign_rec(state, ex_state, lhs_struct, new_rhs, guard,node_id);
}

/*******************************************************************\

Function: basic_symext::symex_assign_if

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_if(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  // we have (c?a:b)=e;

  if(lhs.operands().size()!=3)
    throw "if must have three operands";

  unsigned old_guard_size=guard.size();

  // need to copy rhs -- it gets destroyed
  exprt rhs_copy(rhs);

  exprt condition(lhs.op0());

  guard.add(condition);
  symex_assign_rec(state, ex_state, lhs.op1(), rhs, guard,node_id);
  guard.resize(old_guard_size);

  condition.make_not();

  guard.add(condition);
  symex_assign_rec(state, ex_state, lhs.op2(), rhs_copy, guard,node_id);
  guard.resize(old_guard_size);
}

/*******************************************************************\

Function: basic_symext::symex_assign_byte_extract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::symex_assign_byte_extract(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  // we have byte_extract_X(l, b)=r
  // turn into l=byte_update_X(l, b, r)

  if(lhs.operands().size()!=2)
    throw "byte_extract must have two operands";

  exprt new_rhs;

  if(lhs.id()=="byte_extract_little_endian")
    new_rhs.id("byte_update_little_endian");
  else if(lhs.id()=="byte_extract_big_endian")
    new_rhs.id("byte_update_big_endian");
  else
    assert(false);

  new_rhs.copy_to_operands(lhs.op0(), lhs.op1(), rhs);
  new_rhs.type()=lhs.op0().type();

  symex_assign_rec(state, ex_state, lhs.op0(), new_rhs, guard,node_id);
}

/*******************************************************************\

Function: basic_symext::replace_dynamic_allocation

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::replace_dynamic_allocation(
  const statet &state,
  exprt &expr)
{
  ::replace_dynamic_allocation(ns, expr);
}

/*******************************************************************\

Function: basic_symext::replace_nondet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symext::replace_nondet(exprt &expr)
{
  if(expr.id()=="sideeffect" && expr.get("statement")=="nondet")
  {
    exprt new_expr("nondet_symbol", expr.type());
    new_expr.set("identifier", "symex::nondet"+i2string(nondet_count++));
    new_expr.location()=expr.location();
    expr.swap(new_expr);
  }
  else
    Forall_operands(it, expr)
      replace_nondet(*it);
}

/*******************************************************************\

Function: basic_symex

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symex(
  const codet &code,
  const namespacet &ns,
  symex_targett &target,
  execution_statet &ex_state,
  goto_symex_statet &state,
  unsigned node_id)
{
  contextt new_context;
  basic_symext basic_symex(ns, new_context, target);
  basic_symex.symex(state, ex_state, code, node_id);
}

/*******************************************************************\

Function: basic_symex

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void basic_symex(
  const codet &code,
  const namespacet &ns,
  symex_targett &target,
  execution_statet &ex_state,
        unsigned node_id)
{
  contextt new_context;
  basic_symext basic_symex(ns, new_context, target);
  goto_symex_statet state;
  basic_symex.symex(state, ex_state, code, node_id);
}
