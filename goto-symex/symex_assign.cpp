/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>

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
                         const goto_functionst &_goto_functions,
                         symex_targett *_target, const optionst &opts) :
  guard_identifier_s("goto_symex::\\guard"),
  total_claims(0),
  remaining_claims(0),
  constant_propagation(true),
  ns(_ns),
  options(opts),
  new_context(_new_context),
  goto_functions(_goto_functions),
  target(_target),
  cur_state(NULL)
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

  // Work out whether or not we'll be modelling with cpp:: or c:: arrays.
  const symbolt *sp;
  if (!ns.lookup(irep_idt("c::__ESBMC_alloc"), sp)) {
    valid_ptr_arr_name = "c::__ESBMC_alloc";
    alloc_size_arr_name = "c::__ESBMC_alloc_size";
    deallocd_arr_name = "c::__ESBMC_deallocated";
    dyn_info_arr_name = "c::__ESBMC_is_dynamic";
  } else {
    valid_ptr_arr_name = "cpp::__ESBMC_alloc";
    alloc_size_arr_name = "cpp::__ESBMC_alloc_size";
    deallocd_arr_name = "cpp::__ESBMC_deallocated";
    dyn_info_arr_name = "cpp::__ESBMC_is_dynamic";
  }
}

goto_symext::goto_symext(const goto_symext &sym) :
  ns(sym.ns),
  options(sym.options),
  new_context(sym.new_context),
  goto_functions(sym.goto_functions)
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
  guard_identifier_s = sym.guard_identifier_s;

  valid_ptr_arr_name = sym.valid_ptr_arr_name;
  alloc_size_arr_name = sym.alloc_size_arr_name;
  deallocd_arr_name = sym.deallocd_arr_name;
  dyn_info_arr_name = sym.dyn_info_arr_name;

  // Art ptr is shared
  art1 = sym.art1;

  // Symex target is another matter; a higher up class needs to decide
  // whether we're duplicating it or using the same one.
  target = NULL;

  return *this;
}

void goto_symext::do_simplify(exprt &expr)
{
  if(!options.get_bool_option("no-simplify"))
    simplify(expr);
}

void goto_symext::do_simplify(expr2tc &expr)
{
  if(!options.get_bool_option("no-simplify")) {
    expr2tc tmp = expr->simplify();
    if (!is_nil_expr(tmp))
      expr = tmp;
  }
}

void goto_symext::symex_assign(const codet &code)
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

      if(side_effect_expr.op0().id()!=exprt::symbol)
        throw "symex_assign: expected symbol as function";

      const irep_idt &identifier=
        to_symbol_expr(side_effect_expr.op0()).get_identifier();

      throw "symex_assign: unexpected function call: "+id2string(identifier);
    }
    else if(statement=="cpp_new" ||
            statement=="cpp_new[]") {
      expr2tc new_lhs, tmp_side_effect;
      migrate_expr(lhs, new_lhs);
      migrate_expr(side_effect_expr, tmp_side_effect);
      const sideeffect2t &sideeffect = to_sideeffect2t(tmp_side_effect);
      symex_cpp_new(new_lhs, sideeffect);
    } else if(statement=="malloc") {
      expr2tc new_lhs, tmp_side_effect;
      migrate_expr(lhs, new_lhs);
      migrate_expr(side_effect_expr, tmp_side_effect);
      const sideeffect2t &sideeffect = to_sideeffect2t(tmp_side_effect);
      symex_malloc(new_lhs, sideeffect);
    } else if(statement=="printf") {
      expr2tc new_lhs, new_rhs;
      migrate_expr(lhs, new_lhs);
      migrate_expr(side_effect_expr, new_rhs);
      symex_printf(new_lhs, new_rhs);
    }
    else
    {
      throw "symex_assign: unexpected sideeffect: "+id2string(statement);
    }
  }
  else
  {
    guardt guard; // NOT the state guard!
    expr2tc new_lhs, new_rhs;
    migrate_expr(lhs, new_lhs);
    migrate_expr(rhs, new_rhs);
    symex_assign_rec(new_lhs, new_rhs, guard);
  }
}

void goto_symext::symex_assign_rec(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{

  if (is_symbol2t(lhs)) {
    symex_assign_symbol(lhs, rhs, guard);
  } else if (is_index2t(lhs))
    symex_assign_array(lhs, rhs, guard);
  else if (is_member2t(lhs))
    symex_assign_member(lhs, rhs, guard);
  else if (is_if2t(lhs))
    symex_assign_if(lhs, rhs, guard);
  else if (is_typecast2t(lhs))
    symex_assign_typecast(lhs, rhs, guard);
  else if (is_constant_string2t(lhs) ||
           is_null_object2t(lhs) ||
           is_zero_string2t(lhs))
  {
    // ignore
  }
  else if (is_byte_extract2t(lhs))
    symex_assign_byte_extract(lhs, rhs, guard);
  else
    throw "assignment to " + get_expr_id(lhs) + " not handled";
}

void goto_symext::symex_assign_symbol(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{
  // put assignment guard in rhs

  if (!guard.empty())
  {
    expr2tc guardexpr;
    migrate_expr(guard.as_expr(), guardexpr);
    rhs = expr2tc(new if2t(rhs->type, guardexpr, rhs, lhs));
  }

  expr2tc orig_name_lhs = lhs;
  cur_state->get_original_name(orig_name_lhs);
  cur_state->rename(rhs);

  do_simplify(rhs);

  expr2tc renamed_lhs = lhs;
  cur_state->assignment(renamed_lhs, rhs, constant_propagation);

  guardt tmp_guard(cur_state->guard);
  tmp_guard.append(guard);

  expr2tc guard2;
  migrate_expr(tmp_guard.as_expr(), guard2);

  // do the assignment
  target->assignment(
    guard2,
    renamed_lhs, orig_name_lhs,
    rhs,
    cur_state->source,
    cur_state->gen_stack_trace(),
    symex_targett::STATE);

}

void goto_symext::symex_assign_typecast(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{
  // these may come from dereferencing on the lhs

  const typecast2t &cast = to_typecast2t(lhs);
  expr2tc rhs_typecasted = rhs;
  rhs_typecasted = expr2tc(new typecast2t(cast.from->type, rhs));

  symex_assign_rec(cast.from, rhs_typecasted, guard);
}

void goto_symext::symex_assign_array(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{
  // lhs must be index operand
  // that takes two operands: the first must be an array
  // the second is the index

  const index2t &index = to_index2t(lhs);

  assert(is_array_type(index.source_value->type));

  // turn
  //   a[i]=e
  // into
  //   a'==a WITH [i:=e]

  expr2tc new_rhs = expr2tc(new with2t(index.source_value->type,
                                       index.source_value,
                                       index.index,
                                       rhs));

  symex_assign_rec(index.source_value, new_rhs, guard);
}

void goto_symext::symex_assign_member(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{
  // symbolic execution of a struct member assignment

  // lhs must be member operand
  // that takes one operands, which must be a structure

  const member2t &member = to_member2t(lhs);

  assert(is_struct_type(member.source_value->type) ||
         is_union_type(member.source_value->type));

  const irep_idt &component_name = member.member;
  expr2tc real_lhs = member.source_value;

  // typecasts involved? C++ does that for inheritance.
  if (is_typecast2t(member.source_value))
  {
    const typecast2t &cast = to_typecast2t(member.source_value);
    if (is_null_object2t(cast.from))
    {
      // ignore
    }
    else
    {
      // remove the type cast, we assume that the member is there
      real_lhs = cast.from;
      assert(is_struct_type(real_lhs->type) || is_union_type(real_lhs->type));
    }
  }

  // turn
  //   a.c=e
  // into
  //   a'==a WITH [c:=e]

  type2tc str_type =
    type2tc(new string_type2t(component_name.as_string().size()));
  expr2tc new_rhs = expr2tc(new with2t(real_lhs->type, real_lhs,
                       expr2tc(new constant_string2t(str_type, component_name)),
                       rhs));

  symex_assign_rec(member.source_value, new_rhs, guard);
}

void goto_symext::symex_assign_if(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{
  // we have (c?a:b)=e;

  unsigned old_guard_size=guard.size();

  // need to copy rhs -- it gets destroyed
  expr2tc rhs_copy = rhs;
  const if2t &ifval = to_if2t(lhs);

  expr2tc cond = ifval.cond;

  guard.add(migrate_expr_back(cond));
  symex_assign_rec(ifval.true_value, rhs, guard);
  guard.resize(old_guard_size);

  expr2tc not_cond = expr2tc(new not2t(cond));

  guard.add(migrate_expr_back(not_cond));
  symex_assign_rec(ifval.false_value, rhs_copy, guard);
  guard.resize(old_guard_size);
}

void goto_symext::symex_assign_byte_extract(
  const expr2tc &lhs,
  expr2tc &rhs,
  guardt &guard)
{
  // we have byte_extract_X(l, b)=r
  // turn into l=byte_update_X(l, b, r)

  const byte_extract2t &extract = to_byte_extract2t(lhs);
  expr2tc new_rhs = expr2tc(new byte_update2t(extract.source_value->type,
                                              extract.big_endian,
                                              extract.source_value,
                                              extract.source_offset,
                                              rhs));

  symex_assign_rec(extract.source_value, new_rhs, guard);
}

void goto_symext::replace_nondet(exprt &expr)
{
  if(expr.id()=="sideeffect" && expr.statement()=="nondet")
  {
    unsigned int &nondet_count = get_dynamic_counter();
    exprt new_expr("nondet_symbol", expr.type());
    new_expr.identifier("symex::nondet"+i2string(nondet_count++));
    new_expr.location()=expr.location();
    expr.swap(new_expr);
  }
  else
    Forall_operands(it, expr)
      replace_nondet(*it);
}
