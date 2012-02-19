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

#include "goto_symex.h"
#include "dynamic_allocation.h"
#include "execution_state.h"

goto_symext::goto_symext(const namespacet &_ns, contextt &_new_context,
                         symex_targett *_target, const optionst &opts) :
  total_claims(0),
  remaining_claims(0),
  guard_identifier_s("goto_symex::\\guard"),
  constant_propagation(true),
  ns(_ns),
  options(opts),
  new_context(_new_context),
  target(_target)
{
  const std::string &set = options.get_option("unwindset");
  unsigned int length = set.length();

  for(unsigned int idx = 0; idx < length; idx++)
  {
    std::string::size_type next = set.find(",", idx);
    std::string val = set.substr(idx, next - idx);
    unsigned long id = atoi(val.substr(0, val.find(":", 0)).c_str());
    unsigned long uw = atol(val.substr(val.find(":", 0) + 1).c_str());
    unwind_set[id] = uw;
    if(next == std::string::npos) break;
    idx = next;
  }

  max_unwind=atol(options.get_option("unwind").c_str());

  art1 = NULL;
}

goto_symext::goto_symext(const goto_symext &sym) :
  ns(sym.ns),
  options(sym.options),
  new_context(sym.new_context)
{
  *this = sym;
}

goto_symext& goto_symext::operator=(const goto_symext &sym)
{
  body_warnings = sym.body_warnings;
  unwind_set = sym.unwind_set;
  max_unwind = sym.max_unwind;
  constant_propagation = sym.constant_propagation;
  total_claims = sym.total_claims;
  remaining_claims = sym.remaining_claims;

  // Art ptr is shared
  art1 = sym.art1;

  // Symex target is another matter; a higher up class needs to decide
  // whether we're duplicating it or using the same one.
  target = NULL;
  target = sym.target->clone();

  return *this;
}

/*******************************************************************\

Function: goto_symext::assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::assignment(
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
    state.gen_stack_trace(),
    symex_targett::STATE);
}

/*******************************************************************\

Function: goto_symext::do_simplify

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::do_simplify(exprt &expr)
{
  if(!options.get_bool_option("no-simplify"))
  {
    base_type(expr, ns);
    simplify(expr);
  }
}

/*******************************************************************\

Function: goto_symext::symex_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign(statet &state, execution_statet &ex_state, const codet &code,unsigned node_id)
{
  if(code.operands().size()!=2)
    throw "assignment expects two operands";

  exprt lhs=code.op0();
  exprt rhs=code.op1();

  //replace_dynamic_allocation(state, lhs);
  //replace_dynamic_allocation(state, rhs);

  replace_nondet(lhs, ex_state);
  replace_nondet(rhs, ex_state);

  if(rhs.id()=="sideeffect")
  {
    const side_effect_exprt &side_effect_expr=to_side_effect_expr(rhs);
    const irep_idt &statement=side_effect_expr.get_statement();

    if(statement=="function_call")
    {
      assert(side_effect_expr.operands().size()!=0);

      if(side_effect_expr.op0().id()!=exprt::symbol)
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

Function: goto_symext::symex_assign_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_rec(
  statet &state,
  execution_statet &ex_state,
  const exprt &lhs,
  exprt &rhs,
  guardt &guard,
        unsigned node_id)
{
  if(lhs.id()==exprt::symbol)
    symex_assign_symbol(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()==exprt::index || lhs.id()=="memory-leak")
    symex_assign_array(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()==exprt::member)
    symex_assign_member(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()==exprt::i_if)
    symex_assign_if(state, ex_state, lhs, rhs, guard,node_id);
  else if(lhs.id()==exprt::typecast)
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

Function: goto_symext::symex_assign_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_symbol(
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
    exprt new_rhs(exprt::i_if, rhs.type());
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
    state.gen_stack_trace(),
    symex_targett::STATE);

}

/*******************************************************************\

Function: goto_symext::symex_assign_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_typecast(
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

Function: goto_symext::symex_assign_array

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_array(
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

  if(lhs_type.id()!=typet::t_array)
    throw "index must take array type operand";

  // turn
  //   a[i]=e
  // into
  //   a'==a WITH [i:=e]

  exprt new_rhs(exprt::with, lhs_type);

  new_rhs.reserve_operands(3);
  new_rhs.copy_to_operands(lhs_array);
  new_rhs.copy_to_operands(lhs_index);
  new_rhs.move_to_operands(rhs);

  symex_assign_rec(state, ex_state, lhs_array, new_rhs, guard,node_id);
}

/*******************************************************************\

Function: goto_symext::symex_assign_member

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_member(
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

  if(struct_type.id()!=typet::t_struct &&
     struct_type.id()!=typet::t_union)
    throw "member must take struct/union type operand but got "
          +struct_type.pretty();

  const irep_idt &component_name=lhs.component_name();

  // typecasts involved? C++ does that for inheritance.
  if(lhs_struct.id()==exprt::typecast)
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
      assert(struct_type.id()==typet::t_struct || struct_type.id()==typet::t_union);
      lhs_struct=tmp;
    }
  }

  // turn
  //   a.c=e
  // into
  //   a'==a WITH [c:=e]

  exprt new_rhs(exprt::with, struct_type);

  new_rhs.reserve_operands(3);
  new_rhs.copy_to_operands(lhs_struct);
  new_rhs.copy_to_operands(exprt("member_name"));
  new_rhs.move_to_operands(rhs);

  new_rhs.op1().component_name(component_name);

  symex_assign_rec(state, ex_state, lhs_struct, new_rhs, guard,node_id);
}

/*******************************************************************\

Function: goto_symext::symex_assign_if

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_if(
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

Function: goto_symext::symex_assign_byte_extract

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::symex_assign_byte_extract(
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

Function: goto_symext::replace_nondet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_symext::replace_nondet(exprt &expr, execution_statet &ex_state)
{
  if(expr.id()=="sideeffect" && expr.statement()=="nondet")
  {
    exprt new_expr("nondet_symbol", expr.type());
    new_expr.identifier("symex::nondet"+i2string(ex_state.nondet_count++));
    new_expr.location()=expr.location();
    expr.swap(new_expr);
  }
  else
    Forall_operands(it, expr)
      replace_nondet(*it, ex_state);
}
