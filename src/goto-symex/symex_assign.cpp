/*******************************************************************\

Module: Symbolic Execution

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <boost/shared_ptr.hpp>
#include <cassert>
#include <goto-symex/dynamic_allocation.h>
#include <goto-symex/execution_state.h>
#include <goto-symex/goto_symex.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/simplify_expr.h>
#include <util/std_expr.h>

goto_symext::goto_symext(
  const namespacet &_ns,
  contextt &_new_context,
  const goto_functionst &_goto_functions,
  boost::shared_ptr<symex_targett> _target,
  optionst &opts)
  : options(opts),
    guard_identifier_s("goto_symex::guard"),
    first_loop(0),
    total_claims(0),
    remaining_claims(0),
    max_unwind(options.get_option("unwind").c_str()),
    constant_propagation(!options.get_bool_option("no-propagation")),
    ns(_ns),
    new_context(_new_context),
    goto_functions(_goto_functions),
    target(std::move(_target)),
    cur_state(nullptr),
    last_throw(nullptr),
    inside_unexpected(false),
    no_return_value_opt(options.get_bool_option("no-return-value-opt")),
    stack_limit(atol(options.get_option("stack-limit").c_str())),
    depth_limit(atol(options.get_option("depth").c_str())),
    break_insn(atol(options.get_option("break-at").c_str())),
    memory_leak_check(options.get_bool_option("memory-leak-check")),
    no_assertions(options.get_bool_option("no-assertions")),
    no_simplify(options.get_bool_option("no-simplify")),
    no_unwinding_assertions(options.get_bool_option("no-unwinding-assertions")),
    partial_loops(options.get_bool_option("partial-loops")),
    k_induction(options.get_bool_option("k-induction")),
    base_case(options.get_bool_option("base-case")),
    forward_condition(options.get_bool_option("forward-condition")),
    inductive_step(options.get_bool_option("inductive-step"))
{
  const std::string &set = options.get_option("unwindset");
  unsigned int length = set.length();

  for(unsigned int idx = 0; idx < length; idx++)
  {
    std::string::size_type next = set.find(",", idx);
    std::string val = set.substr(idx, next - idx);
    unsigned long id = atoi(val.substr(0, val.find(":", 0)).c_str());
    BigInt uw(val.substr(val.find(":", 0) + 1).c_str());
    unwind_set[id] = uw;
    if(next == std::string::npos)
      break;
    idx = next;
  }

  art1 = nullptr;

  valid_ptr_arr_name = "c:@__ESBMC_alloc";
  alloc_size_arr_name = "c:@__ESBMC_alloc_size";
  deallocd_arr_name = "c:@__ESBMC_deallocated";
  dyn_info_arr_name = "c:@__ESBMC_is_dynamic";

  symbolt sym;
  sym.id = "symex_throw::thrown_obj";
  sym.name = "thrown_obj";
  // Type left deliberately undefined. XXX, is this wise?
  new_context.move(sym);
}

goto_symext::goto_symext(const goto_symext &sym)
  : options(sym.options),
    ns(sym.ns),
    new_context(sym.new_context),
    goto_functions(sym.goto_functions),
    last_throw(nullptr),
    inside_unexpected(false)
{
  *this = sym;
}

goto_symext &goto_symext::operator=(const goto_symext &sym)
{
  unwind_set = sym.unwind_set;
  max_unwind = sym.max_unwind;
  constant_propagation = sym.constant_propagation;
  total_claims = sym.total_claims;
  remaining_claims = sym.remaining_claims;
  guard_identifier_s = sym.guard_identifier_s;
  depth_limit = sym.depth_limit;
  break_insn = sym.break_insn;
  memory_leak_check = sym.memory_leak_check;
  no_assertions = sym.no_assertions;
  no_simplify = sym.no_simplify;
  no_unwinding_assertions = sym.no_unwinding_assertions;
  partial_loops = sym.partial_loops;
  k_induction = sym.k_induction;
  base_case = sym.base_case;
  forward_condition = sym.forward_condition;
  inductive_step = sym.inductive_step;
  first_loop = sym.first_loop;

  valid_ptr_arr_name = sym.valid_ptr_arr_name;
  alloc_size_arr_name = sym.alloc_size_arr_name;
  deallocd_arr_name = sym.deallocd_arr_name;
  dyn_info_arr_name = sym.dyn_info_arr_name;

  dynamic_memory = sym.dynamic_memory;

  // Art ptr is shared
  art1 = sym.art1;

  // Symex target is another matter; a higher up class needs to decide
  // whether we're duplicating it or using the same one.
  target = nullptr;

  return *this;
}

void goto_symext::do_simplify(expr2tc &expr)
{
  if(!no_simplify)
    simplify(expr);
}

void goto_symext::symex_assign(
  const expr2tc &code_assign,
  const bool hidden,
  const guardt &guard)
{
  const code_assign2t &code = to_code_assign2t(code_assign);

  // Sanity check: if the target has zero size, then we've ended up assigning
  // to/from a C++ POD class with no fields. The rest of the model checker isn't
  // rated for dealing with this concept; perform a NOP.
  try
  {
    if(is_struct_type(code.target->type))
    {
      const struct_type2t &t2 =
        static_cast<const struct_type2t &>(*code.target->type);

      if(!t2.members.size())
        return;
    }
  }
  catch(array_type2t::dyn_sized_array_excp *foo)
  {
    delete foo;
  }

  expr2tc original_lhs = code.target;
  expr2tc lhs = code.target;
  expr2tc rhs = code.source;

  replace_nondet(lhs);
  replace_nondet(rhs);

  dereference(lhs, dereferencet::WRITE);
  dereference(rhs, dereferencet::READ);
  replace_dynamic_allocation(lhs);
  replace_dynamic_allocation(rhs);

  if(is_sideeffect2t(rhs))
  {
    const sideeffect2t &effect = to_sideeffect2t(rhs);
    switch(effect.kind)
    {
    case sideeffect2t::cpp_new:
    case sideeffect2t::cpp_new_arr:
      symex_cpp_new(lhs, effect);
      break;
    case sideeffect2t::realloc:
      symex_realloc(lhs, effect);
      break;
    case sideeffect2t::malloc:
      symex_malloc(lhs, effect);
      break;
    case sideeffect2t::alloca:
      symex_alloca(lhs, effect);
      break;
    case sideeffect2t::va_arg:
      symex_va_arg(lhs, effect);
      break;
    // No nondet side effect?
    default:
      assert(0 && "unexpected side effect");
    }

    return;
  }

  bool hidden_ssa = hidden || cur_state->top().hidden;
  if(!hidden_ssa)
  {
    auto const maybe_symbol = get_base_object(lhs);
    if(is_symbol2t(maybe_symbol))
    {
      auto const s = to_symbol2t(maybe_symbol).thename.as_string();
      hidden_ssa |= (s.find('$') != std::string::npos) ||
                    (s.find("__ESBMC_") != std::string::npos);
    }
  }

  guardt g(guard); // NOT the state guard!
  symex_assign_rec(lhs, original_lhs, rhs, g, hidden_ssa);
}

void goto_symext::symex_assign_rec(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  if(is_symbol2t(lhs))
  {
    symex_assign_symbol(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_index2t(lhs))
  {
    symex_assign_array(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_member2t(lhs))
  {
    symex_assign_member(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_if2t(lhs))
  {
    symex_assign_if(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_typecast2t(lhs) || is_bitcast2t(lhs))
  {
    symex_assign_typecast(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_constant_string2t(lhs) || is_null_object2t(lhs))
  {
    // ignore
  }
  else if(is_byte_extract2t(lhs))
  {
    symex_assign_byte_extract(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_concat2t(lhs))
  {
    symex_assign_concat(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_constant_struct2t(lhs))
  {
    symex_assign_structure(lhs, full_lhs, rhs, guard, hidden);
  }
  else if(is_extract2t(lhs))
  {
    symex_assign_extract(lhs, full_lhs, rhs, guard, hidden);
  }
  else
  {
    std::cerr << "assignment to " << get_expr_id(lhs) << " not handled\n";
    abort();
  }
}

void goto_symext::symex_assign_symbol(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  // put assignment guard in rhs
  if(!guard.is_true())
    rhs = if2tc(rhs->type, guard.as_expr(), rhs, lhs);

  cur_state->rename(rhs);
  do_simplify(rhs);

  expr2tc renamed_lhs = lhs;
  cur_state->rename_type(renamed_lhs);
  cur_state->assignment(renamed_lhs, rhs);

  // Special case when the lhs is an array access, we need to get the
  // right symbol for the index
  expr2tc new_lhs = full_lhs;
  if(is_index2t(new_lhs))
    cur_state->rename(to_index2t(new_lhs).index);

  guardt tmp_guard(cur_state->guard);
  tmp_guard.append(guard);

  // do the assignment
  target->assignment(
    tmp_guard.as_expr(),
    renamed_lhs,
    new_lhs,
    rhs,
    cur_state->source,
    cur_state->gen_stack_trace(),
    hidden,
    first_loop);
}

void goto_symext::symex_assign_structure(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  const struct_type2t &structtype = to_struct_type(lhs->type);
  const constant_struct2t &the_structure = to_constant_struct2t(lhs);

  // Explicitly project lhs fields out of structure, assignment will just undo
  // any member operations. If user is assigning to a structure literal, we
  // will croak after recursing. Otherwise, we are assigning to a re-constituted
  // structure, through dereferencing.
  unsigned int i = 0;
  for(auto const &it : structtype.members)
  {
    const expr2tc &lhs_memb = the_structure.datatype_members[i];
    member2tc rhs_memb(it, rhs, structtype.member_names[i]);
    symex_assign_rec(lhs_memb, full_lhs, rhs_memb, guard, hidden);
    i++;
  }
}

void goto_symext::symex_assign_typecast(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  // these may come from dereferencing on the lhs

  const typecast_data &cast = dynamic_cast<const typecast_data &>(*lhs.get());
  expr2tc rhs_typecasted = rhs;
  if(is_typecast2t(lhs))
  {
    rhs_typecasted = typecast2tc(cast.from->type, rhs);
  }
  else
  {
    assert(is_bitcast2t(lhs));
    rhs_typecasted = bitcast2tc(cast.from->type, rhs);
  }

  symex_assign_rec(cast.from, full_lhs, rhs_typecasted, guard, hidden);
}

void goto_symext::symex_assign_array(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  // lhs must be index operand
  // that takes two operands: the first must be an array
  // the second is the index

  const index2t &index = to_index2t(lhs);

  assert(
    is_array_type(index.source_value) || is_string_type(index.source_value));

  // turn
  //   a[i]=e
  // into
  //   a'==a WITH [i:=e]

  with2tc new_rhs(
    index.source_value->type, index.source_value, index.index, rhs);

  symex_assign_rec(index.source_value, full_lhs, new_rhs, guard, hidden);
}

void goto_symext::symex_assign_member(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  // symbolic execution of a struct member assignment

  // lhs must be member operand
  // that takes one operands, which must be a structure

  const member2t &member = to_member2t(lhs);

  assert(
    is_struct_type(member.source_value) || is_union_type(member.source_value));

  const irep_idt &component_name = member.member;
  expr2tc real_lhs = member.source_value;

  // typecasts involved? C++ does that for inheritance.
  if(is_typecast2t(member.source_value))
  {
    const typecast2t &cast = to_typecast2t(member.source_value);
    if(is_null_object2t(cast.from))
    {
      // ignore
    }
    else
    {
      // remove the type cast, we assume that the member is there
      real_lhs = cast.from;
      assert(is_struct_type(real_lhs) || is_union_type(real_lhs));
    }
  }

  // turn
  //   a.c=e
  // into
  //   a'==a WITH [c:=e]

  type2tc str_type =
    type2tc(new string_type2t(component_name.as_string().size()));
  with2tc new_rhs(
    real_lhs->type,
    real_lhs,
    constant_string2tc(str_type, component_name),
    rhs);

  symex_assign_rec(member.source_value, full_lhs, new_rhs, guard, hidden);
}

void goto_symext::symex_assign_if(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  // we have (c?a:b)=e;

  // need to copy rhs -- it gets destroyed
  expr2tc rhs_copy = rhs;
  const if2t &ifval = to_if2t(lhs);

  expr2tc cond = ifval.cond;

  guardt old_guard(guard);

  guard.add(cond);
  symex_assign_rec(ifval.true_value, full_lhs, rhs, guard, hidden);
  guard = old_guard;

  not2tc not_cond(cond);
  guard.add(not_cond);
  symex_assign_rec(ifval.false_value, full_lhs, rhs_copy, guard, hidden);
  guard = old_guard;
}

void goto_symext::symex_assign_byte_extract(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  // we have byte_extract_X(l, b)=r
  // turn into l=byte_update_X(l, b, r)

  // Grief: multi dimensional arrays.
  const byte_extract2t &extract = to_byte_extract2t(lhs);

  if(is_multi_dimensional_array(extract.source_value))
  {
    const array_type2t &arr_type = to_array_type(extract.source_value->type);
    assert(
      !is_multi_dimensional_array(arr_type.subtype) &&
      "Can't currently byte extract through more than two dimensions of "
      "array right now, sorry");
    constant_int2tc subtype_sz(index_type2(), type_byte_size(arr_type.subtype));
    expr2tc div = div2tc(index_type2(), extract.source_offset, subtype_sz);
    expr2tc mod = modulus2tc(index_type2(), extract.source_offset, subtype_sz);
    do_simplify(div);
    do_simplify(mod);

    index2tc idx(arr_type.subtype, extract.source_value, div);
    byte_update2tc be2(arr_type.subtype, idx, mod, rhs, extract.big_endian);
    with2tc store(extract.source_value->type, extract.source_value, div, be2);
    symex_assign_rec(extract.source_value, full_lhs, store, guard, hidden);
  }
  else
  {
    byte_update2tc new_rhs(
      extract.source_value->type,
      extract.source_value,
      extract.source_offset,
      rhs,
      extract.big_endian);

    symex_assign_rec(extract.source_value, full_lhs, new_rhs, guard, hidden);
  }
}

void goto_symext::symex_assign_concat(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
// Right: generate a series of symex assigns.
#ifndef NDEBUG
  const concat2t &cat = to_concat2t(lhs);
  assert(cat.type->get_width() > 8);
#endif
  assert(is_scalar_type(rhs));

  // Second attempt at this code: byte stitching guarantees that all the concats
  // occur in one large grouping. Produce a list of them.
  std::list<expr2tc> operand_list;
  expr2tc cur_concat = lhs;
  while(is_concat2t(cur_concat))
  {
    const concat2t &cat2 = to_concat2t(cur_concat);
    operand_list.push_back(cat2.side_2);
    cur_concat = cat2.side_1;
  }

  // Add final operand to list
  operand_list.push_back(cur_concat);

#ifndef NDEBUG
  for(auto const &foo : operand_list)
    assert(foo->type->get_width() == 8);
#endif
  assert((operand_list.size() * 8) == cat.type->get_width());

  bool is_big_endian =
    (config.ansi_c.endianess == configt::ansi_ct::IS_BIG_ENDIAN);

  // Pin one set of rhs version numbers: if we assign part of a value to itself,
  // it'll change during the assignment
  cur_state->rename(rhs);

  // Produce a corresponding set of byte extracts from the rhs value. Note that
  // the byte offset is always the same no matter endianness here, any byte
  // order flipping is handled at the smt layer.
  std::list<expr2tc> extracts;
  for(unsigned int i = 0; i < operand_list.size(); i++)
  {
    byte_extract2tc byte(get_uint_type(8), rhs, gen_ulong(i), is_big_endian);
    extracts.push_back(byte);
  }

  // Now proceed to pair them up
  assert(extracts.size() == operand_list.size());
  auto lhs_it = operand_list.begin();
  auto rhs_it = extracts.begin();
  while(lhs_it != operand_list.end())
  {
    symex_assign_rec(*lhs_it, full_lhs, *rhs_it, guard, hidden);
    lhs_it++;
    rhs_it++;
  }
}

void goto_symext::symex_assign_extract(
  const expr2tc &lhs,
  const expr2tc &full_lhs,
  expr2tc &rhs,
  guardt &guard,
  const bool hidden)
{
  const extract2t &ex = to_extract2t(lhs);
  assert(is_bv_type(ex.from));
  assert(is_bv_type(rhs));
  assert(rhs->type->get_width() == lhs->type->get_width());

  // We need to: read the rest of the bitfield and reconstruct it. Extract
  // and concats are probably the best approach for the solver to optimise for.
  unsigned int bitblob_width = ex.from->type->get_width();
  expr2tc top_part;
  if(ex.upper != bitblob_width - 1)
  {
    // Extract from the top of the blob down to the bit above this extract
    type2tc thetype = get_uint_type(bitblob_width - ex.upper - 1);
    top_part = extract2tc(thetype, ex.from, bitblob_width - 1, ex.upper + 1);
  }

  expr2tc bottom_part;
  if(ex.lower != 0)
  {
    type2tc thetype = get_uint_type(ex.lower);
    bottom_part = extract2tc(thetype, ex.from, ex.lower - 1, 0);
  }

  // We now have two or three parts: accumulate them into a bitblob sized lump
  expr2tc accuml = bottom_part;
  if(!is_nil_expr(accuml))
  {
    type2tc thetype =
      get_uint_type(accuml->type->get_width() + rhs->type->get_width());
    accuml = concat2tc(thetype, rhs, bottom_part);
  }
  else
  {
    accuml = rhs;
  }

  if(!is_nil_expr(top_part))
  {
    assert(
      accuml->type->get_width() + top_part->type->get_width() == bitblob_width);
    type2tc thetype = get_uint_type(bitblob_width);
    accuml = concat2tc(thetype, top_part, accuml);
  }
  else
  {
    assert(accuml->type->get_width() == bitblob_width);
  }

  // OK: accuml now has a bitblob sized expression that can be assigned into
  // the relevant field.
  symex_assign_rec(ex.from, full_lhs, accuml, guard, hidden);
}

void goto_symext::replace_nondet(expr2tc &expr)
{
  if(
    is_sideeffect2t(expr) && to_sideeffect2t(expr).kind == sideeffect2t::nondet)
  {
    unsigned int &nondet_count = get_dynamic_counter();
    expr =
      symbol2tc(expr->type, "nondet$symex::nondet" + i2string(nondet_count++));
  }
  else
  {
    expr->Foreach_operand([this](expr2tc &e) {
      if(!is_nil_expr(e))
        replace_nondet(e);
    });
  }
}
