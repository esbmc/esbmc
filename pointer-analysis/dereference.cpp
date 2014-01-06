/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <assert.h>
#include <sstream>
#include <prefix.h>
#include <expr_util.h>
#include <c_misc.h>
#include <base_type.h>
#include <arith_tools.h>
#include <rename.h>
#include <i2string.h>
#include <array_name.h>
#include <config.h>
#include <std_expr.h>
#include <cprover_prefix.h>
#include <type_byte_size.h>

#include <ansi-c/c_types.h>
#include <ansi-c/c_typecast.h>
#include <pointer-analysis/value_set.h>
#include <langapi/language_util.h>

#include "dereference.h"

// global data, horrible
unsigned int dereferencet::invalid_counter=0;

static inline bool is_non_scalar_expr(const expr2tc &e)
{
  return is_member2t(e) || is_index2t(e) || (is_if2t(e) && !is_scalar_type(e));
}

static inline const array_type2t
get_arr_type(const expr2tc &expr)
{
  return (is_array_type(expr)) ? to_array_type(expr->type)
    : to_array_type(to_constant_string2t(expr).to_array()->type);
}

static inline const type2tc
get_arr_subtype(const expr2tc &expr)
{
  return get_arr_type(expr).subtype;
}

// Look for the base of an expression such as &a->b[1];, where all we're doing
// is performing some pointer arithmetic, rather than actually performing some
// dereference operation.
static inline expr2tc get_base_dereference(const expr2tc &e)
{

  // XXX -- do we need to consider if2t's? And how?
  // XXX -- This doesn't consider indexes of pointers to be dereferences, yet
  // that's how they're handled.
  if (is_member2t(e)) {
    return get_base_dereference(to_member2t(e).source_value);
  } else if (is_index2t(e) && is_pointer_type(to_index2t(e).source_value)) {
    return e;
  } else if (is_index2t(e)) {
    return get_base_dereference(to_index2t(e).source_value);
  } else if (is_dereference2t(e)) {
    return to_dereference2t(e).value;
  } else {
    return expr2tc();
  }
}

bool dereferencet::has_dereference(const expr2tc &expr) const
{
  if (is_nil_expr(expr))
    return false;

  forall_operands2(it, idx, expr)
    if(has_dereference(*it))
      return true;

  if (is_dereference2t(expr) ||
     (is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value)))
    return true;

  return false;
}

const expr2tc& dereferencet::get_symbol(const expr2tc &expr)
{
  if (is_member2t(expr))
    return get_symbol(to_member2t(expr).source_value);
  else if (is_index2t(expr))
    return get_symbol(to_index2t(expr).source_value);

  return expr;
}

/************************* Expression decomposing code ************************/

enum expr_deref_handler {
  deref_recurse = 0,
  deref_munge_guard,
  deref_addrof,
  deref_deref,
  deref_nonscalar
};

static char deref_expr_handler_actions[expr2t::end_expr_id];

void
dereference_handlers_init(void)
{
  deref_expr_handler_actions[expr2t::and_id] = deref_munge_guard;
  deref_expr_handler_actions[expr2t::or_id] = deref_munge_guard;
  deref_expr_handler_actions[expr2t::if_id] = deref_munge_guard;
  deref_expr_handler_actions[expr2t::address_of_id] = deref_addrof;
  deref_expr_handler_actions[expr2t::dereference_id] = deref_deref;
  deref_expr_handler_actions[expr2t::index_id] = deref_nonscalar;
  deref_expr_handler_actions[expr2t::member_id] = deref_nonscalar;
}

void
dereferencet::dereference_expr(
  expr2tc &expr,
  guardt &guard,
  modet mode)
{

  if (!has_dereference(expr))
    return;

  switch (deref_expr_handler_actions[expr->expr_id]) {
  case deref_recurse:
  {
    Forall_operands2(it, idx, expr) {
      if (is_nil_expr(*it))
        continue;

      dereference_expr(*it, guard, mode);
    }
    break;
  }
  case deref_munge_guard:
    dereference_guard_expr(expr, guard, mode);
    break;
  case deref_addrof:
    dereference_addrof_expr(expr, guard, mode);
    break;
  case deref_deref:
    dereference_deref(expr, guard, mode);
    break;
  case deref_nonscalar:
  {
    // The result of this expression should be scalar: we're transitioning
    // from a scalar result to a nonscalar result.

    std::list<expr2tc> scalar_step_list;
    expr2tc res = dereference_expr_nonscalar(expr, guard, mode,
                                             scalar_step_list);
    assert(scalar_step_list.size() == 0); // Should finish empty.

    // If a dereference successfully occurred, replace expr at this level.
    // XXX -- explain this better.
    if (!is_nil_expr(res))
      expr = res;
    break;
  }
  }

  return;
}

void
dereferencet::dereference_guard_expr(expr2tc &expr, guardt &guard, modet mode)
{
  if (is_and2t(expr) || is_or2t(expr))
  {
    // If this is an and or or expression, then if the first operand short
    // circuits the truth of the expression, then we shouldn't evaluate the
    // second expression. That means that the dereference assertions in the
    // 2nd should be guarded with the fact that the first operand didn't short
    // circuit.
    assert(is_bool_type(expr));
    // Take the current size of the guard, so that we can reset it later.
    unsigned old_guards=guard.size();

    Forall_operands2(it, idx, expr) {
      expr2tc &op = *it;

      assert(is_bool_type(op));

      // Handle any derererences in this operand
      if (has_dereference(op))
        dereference_expr(op, guard, dereferencet::READ);

      // Guard the next operand against this operand short circuiting us.
      if (is_or2t(expr)) {
        not2tc tmp(op);
        guard.move(tmp);
      } else {
        guard.add(op);
      }
    }

    // Reset guard to where it was.
    guard.resize(old_guards);
    return;
  }
  else
  {
    assert(is_if2t(expr));
    // Only one side of this if gets evaluated according to the condition, which
    // means that pointer dereference assertion failures should have the
    // relevant guard applied. This makes sure they don't fire even when their
    // expression isn't evaluated.
    if2t &ifref = to_if2t(expr);
    dereference_expr(ifref.cond, guard, dereferencet::READ);

    bool o1 = has_dereference(ifref.true_value);
    bool o2 = has_dereference(ifref.false_value);

    if (o1) {
      unsigned old_guard=guard.size();
      guard.add(ifref.cond);
      dereference_expr(ifref.true_value, guard, mode);
      guard.resize(old_guard);
    }

    if (o2) {
      unsigned old_guard=guard.size();
      not2tc tmp(ifref.cond);
      guard.move(tmp);
      dereference_expr(ifref.false_value, guard, mode);
      guard.resize(old_guard);
    }

    return;
  }
}

void
dereferencet::dereference_addrof_expr(expr2tc &expr, guardt &guard, modet mode)
{
  // Crazy combinations of & and * that don't actually lead to a deref:

  // turn &*p to p
  // this has *no* side effect!
  // XXX jmorse -- how does this take account of an intervening member
  // operation? i.e. &foo->bar;
  address_of2t &addrof = to_address_of2t(expr);

  if (is_dereference2t(addrof.ptr_obj)) {
    dereference2t &deref = to_dereference2t(addrof.ptr_obj);
    expr2tc result = deref.value;

    if (result->type != expr->type)
      result = typecast2tc(expr->type, result);

    expr = result;
  } else {
    // This might, alternately, be a chain of member and indexes applied to
    // a dereference. In which case what we're actually doing is computing
    // some pointer arith, manually.
    expr2tc base = get_base_dereference(addrof.ptr_obj);
    if (!is_nil_expr(base)) {
      //  We have a base. There may be additional dereferences in it.
      dereference_expr(base, guard, mode);
      // Now compute the pointer offset involved.
      expr2tc offs = compute_pointer_offset(addrof.ptr_obj);
      assert(!is_nil_expr(offs) && "Pointer offset of index/member "
             "combination should be valid int");

      // Cast to a byte pointer; add; cast back. Can't think of a better way
      // to produce safe pointer arithmetic right now.
      expr2tc output =
        typecast2tc(type2tc(new pointer_type2t(get_uint8_type())), base);
      output = add2tc(output->type, output, offs);
      output = typecast2tc(expr->type, output);
      expr = output;
    } else {
      // It's not something that we can simplify from &foo->bar[baz] to not have
      // a dereference, but might still contain a dereference.
      dereference_expr(addrof.ptr_obj, guard, mode);
    }
  }

  // We modified this expression, but, we might have injected some pointer
  // arithmetic that contains another dereference. So we need to re-deref this
  // new expression.
  dereference_expr(expr, guard, mode);
}

void
dereferencet::dereference_deref(expr2tc &expr, guardt &guard, modet mode)
{
  if (is_dereference2t(expr)) {
    std::list<expr2tc> scalar_step_list;

    dereference2t &deref = to_dereference2t(expr);
    // first make sure there are no dereferences in there
    dereference_expr(deref.value, guard, dereferencet::READ);

    if (is_array_type(to_pointer_type(deref.value->type).subtype)) {
      // Dereferencing yeilding an array means we're actually performing pointer
      // arithmetic, on a multi-dimensional array. The operand is performing
      // said arith. Simply drop this dereference, and massage the type.
      expr2tc tmp = deref.value;
      const array_type2t &arr =
        to_array_type(to_pointer_type(deref.value->type).subtype);

      tmp.get()->type = type2tc(new pointer_type2t(arr.subtype));
      expr = tmp;
//XXX -- test this! nonscalar handles this now?
      return;
    }

    expr2tc tmp_obj = deref.value;
    expr2tc result = dereference(tmp_obj, deref.type, guard, mode,
                                 &scalar_step_list);
    expr = result;
  }
  else
  {
    assert(is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value));
    std::list<expr2tc> scalar_step_list;
    assert((is_scalar_type(expr) || is_code_type(expr))
           && "Can't dereference to a nonscalar type");
    index2t &idx = to_index2t(expr);

    // first make sure there are no dereferences in there
    dereference_expr(idx.index, guard, dereferencet::READ);
    dereference_expr(idx.source_value, guard, mode);

    add2tc tmp(idx.source_value->type, idx.source_value, idx.index);
    // Result discarded.
    expr = dereference(tmp, tmp->type, guard, mode, &scalar_step_list);
  }
}

expr2tc
dereferencet::dereference_expr_nonscalar(
  expr2tc &expr,
  guardt &guard,
  modet mode,
  std::list<expr2tc> &scalar_step_list)
{

  if (is_dereference2t(expr))
  {
    dereference2t &deref = to_dereference2t(expr);
    // first make sure there are no dereferences in there
    dereference_expr(deref.value, guard, dereferencet::READ);
    expr2tc result = dereference(deref.value, type2tc(), guard, mode,
                                 &scalar_step_list);
    return result;
  }
  else if (is_index2t(expr) && is_pointer_type(to_index2t(expr).source_value))
  {
    index2t &index = to_index2t(expr);

    // first make sure there are no dereferences in there
    dereference_expr(index.source_value, guard, dereferencet::READ);
    dereference_expr(index.index, guard, dereferencet::READ);

    add2tc tmp(index.source_value->type, index.source_value, index.index);
    expr2tc result = dereference(tmp, type2tc(), guard, mode,
                                 &scalar_step_list);
    return result;
  }
  else if (is_non_scalar_expr(expr))
  {
    expr2tc res;
    if (is_member2t(expr)) {
      scalar_step_list.push_front(expr);
      res =  dereference_expr_nonscalar(to_member2t(expr).source_value, guard,
                                        mode, scalar_step_list);
      scalar_step_list.pop_front();
    } else if (is_index2t(expr)) {
      dereference_expr(to_index2t(expr).index, guard, dereferencet::READ);
      scalar_step_list.push_front(expr);
      res = dereference_expr_nonscalar(to_index2t(expr).source_value, guard,
                                       mode, scalar_step_list);
      scalar_step_list.pop_front();
    } else if (is_if2t(expr)) {
      guardt g1 = guard, g2 = guard;
      if2t &theif = to_if2t(expr);
      g1.add(theif.cond);
      g2.add(not2tc(theif.cond));

      scalar_step_list.push_front(theif.true_value);
      expr2tc res1 = dereference_expr_nonscalar(theif.true_value, g1, mode,
                                                scalar_step_list);
      scalar_step_list.pop_front();

      scalar_step_list.push_front(theif.false_value);
      expr2tc res2 = dereference_expr_nonscalar(theif.false_value, g2, mode,
                                                scalar_step_list);
      scalar_step_list.pop_front();

      if2tc fin(res1->type, theif.cond, res1, res2);
      res = fin;
    } else {
      std::cerr << "Unexpected expression in dereference_expr_nonscalar"
                << std::endl;
      expr->dump();
      abort();
    }

    return res;
  }
  else if (is_typecast2t(expr))
  {
    // Just blast straight through
    return dereference_expr_nonscalar(to_typecast2t(expr).from, guard, mode,
                                      scalar_step_list);
  }
  else
  {
    // This should end up being either a constant or a symbol; either way
    // there should be no sudden transition back to scalars, except through
    // dereferences. Return nil to indicate that there was no dereference at
    // the bottom of this.
    assert(!is_scalar_type(expr) &&
           (is_constant_expr(expr) || is_symbol2t(expr)));
    return expr2tc();
  }
}

/********************** Intermediate reference munging code *******************/

expr2tc
dereferencet::dereference(
  const expr2tc &src,
  const type2tc &to_type,
  const guardt &guard,
  modet mode,
  std::list<expr2tc> *scalar_step_list)
{
  assert(is_pointer_type(src));
  internal_items.clear();

  // Target type is either a scalar type passed down to us, or we have a chain
  // of scalar steps available that end up at a scalar type. The result of this
  // dereference should be a scalar, via whatever means.
  type2tc type = (!is_nil_type(to_type))
    ? to_type : scalar_step_list->back()->type;

  // collect objects dest may point to
  value_setst::valuest points_to_set;

  dereference_callback.get_value_set(src, points_to_set);

  // now build big case split
  // only "good" objects

  expr2tc value;

  for(value_setst::valuest::const_iterator
      it=points_to_set.begin();
      it!=points_to_set.end();
      it++)
  {
    expr2tc new_value, pointer_guard;

    new_value = build_reference_to(*it, mode, src, type, guard,
                                   scalar_step_list, pointer_guard);

    if (!is_nil_expr(new_value))
    {
      if (is_nil_expr(value)) {
        value = new_value;
      } else {
        // Chain a big if-then-else case.
        value = if2tc(type, pointer_guard, new_value, value);
      }
    }
  }

  if (is_nil_expr(value) && mode != INTERNAL)
  {
    // Dereference failed entirely; various assertions will explode later down
    // the line. To make this a valid formula though, return a failed symbol,
    // so that this assignment gets a well typed free value.
    value = make_failed_symbol(type);
  } else if (mode == INTERNAL) {
    // Deposit internal values with the caller, then clear.
    dereference_callback.dump_internal_state(internal_items);
    internal_items.clear();
  }

  return value;
}

expr2tc
dereferencet::make_failed_symbol(const type2tc &out_type)
{
  // else, do new symbol
  symbolt symbol;
  symbol.name="symex::invalid_object"+i2string(invalid_counter++);
  symbol.base_name="invalid_object";
  symbol.type=migrate_type_back(out_type);

  // make it a lvalue, so we can assign to it
  symbol.lvalue=true;

  get_new_name(symbol, ns);

  exprt tmp_sym_expr = symbol_expr(symbol);

  new_context.move(symbol);

  // Due to migration hiccups, migration must occur after the symbol
  // appears in the symbol table.
  expr2tc value;
  migrate_expr(tmp_sym_expr, value);
  return value;
}

bool dereferencet::dereference_type_compare(
  expr2tc &object, const type2tc &dereference_type) const
{
  const type2tc object_type = object->type;

  // Test for simple equality
  if (object->type == dereference_type)
    return true;

  // Check for C++ subclasses; we can cast derived up to base safely.
  if (is_struct_type(object) && is_struct_type(dereference_type)) {
    if (is_subclass_of(object->type, dereference_type, ns)) {
      object = typecast2tc(dereference_type, object);
      return true;
    }
  }

  // check for struct prefixes

  type2tc ot_base(object_type), dt_base(dereference_type);

  base_type(ot_base, ns);
  base_type(dt_base, ns);

  if (is_struct_type(ot_base) && is_struct_type(dt_base))
  {
    typet tmp_ot_base = migrate_type_back(ot_base);
    typet tmp_dt_base = migrate_type_back(dt_base);
    if (to_struct_type(tmp_dt_base).is_prefix_of(
         to_struct_type(tmp_ot_base)))
    {
      object = typecast2tc(dereference_type, object);
      return true; // ok, dt is a prefix of ot
    }
  }

  // XXX - are there such things as compatible unions?

  // really different

  return false;
}

expr2tc
dereferencet::build_reference_to(
  const expr2tc &what,
  modet mode,
  const expr2tc &deref_expr,
  const type2tc &type,
  const guardt &guard,
  std::list<expr2tc> *scalar_step_list,
  expr2tc &pointer_guard)
{
  expr2tc value;
  pointer_guard = false_expr;

  if (is_unknown2t(what) || is_invalid2t(what))
  {
    // constraint that it actually is an invalid pointer

    invalid_pointer2tc invalid_pointer_expr(deref_expr);

    // produce new guard

    guardt tmp_guard(guard);
    tmp_guard.move(invalid_pointer_expr);
    dereference_failure("pointer dereference", "invalid pointer", tmp_guard);

    return value;
  }

  if (!is_object_descriptor2t(what)) {
    std::cerr << "unknown points-to: " << get_expr_id(what);
    abort();
  }

  const object_descriptor2t &o = to_object_descriptor2t(what);

  const expr2tc &root_object = o.get_root_object();
  const expr2tc &object = o.object;

  if (is_null_object2t(root_object))
  {
    type2tc nullptrtype = type2tc(new pointer_type2t(type));
    symbol2tc null_ptr(nullptrtype, "NULL");

    same_object2tc pointer_guard(deref_expr, null_ptr);

    guardt tmp_guard(guard);
    tmp_guard.add(pointer_guard);

    dereference_failure("pointer dereference", "NULL pointer", tmp_guard);

    // Don't build a reference to this. You can't actually access NULL, and the
    // solver will only get confused.
    return value;
  }

  value = object;

  // Produce a guard that the dererferenced pointer points at this object.
  type2tc ptr_type = type2tc(new pointer_type2t(object->type));
  address_of2tc obj_ptr(ptr_type, object);
  pointer_guard = same_object2tc(deref_expr, obj_ptr);
  guardt tmp_guard(guard);
  tmp_guard.add(pointer_guard);

  // Check that the object we're accessing is actually alive and valid for this
  // mode.
  valid_check(object, tmp_guard, mode);

  // Don't do anything further if we're freeing things
  if (mode == FREE)
    return expr2tc();

  // Try to pull additional offset out of the reference, i.e., member and index
  // expressions. XXX does this make any difference, surely the offset is in
  // the offset field.
  expr2tc additional_offset = compute_pointer_offset(value);
  expr2tc add = add2tc(o.offset->type, o.offset, additional_offset);
#if 0
  // FIXME: benchmark this, on tacas.
  dereference_callback.rename(add);
#endif
  expr2tc final_offset = add->simplify();
  if (is_nil_expr(final_offset))
    final_offset = add;

  // Finally, construct a reference against the base object. value set tracking
  // emits objects with some cruft built on top of them.
  value = get_base_object(value);

  // If offset is unknown, or whatever, instead compute the pointer offset
  // manually.
  if (!is_constant_int2t(final_offset)) {
    final_offset = pointer_offset2tc(index_type2(), deref_expr);
    assert(o.alignment != 0);
  }

  // If we're in internal mode, collect all of our data into one struct, insert
  // it into the list of internal data, and then bail. The caller does not want
  // to have a reference built at all.
  if (mode == INTERNAL) {
    dereference_callbackt::internal_item internal;
    internal.object = value;
    internal.offset = final_offset;
    internal.guard = pointer_guard;
    internal_items.push_back(internal);
    return expr2tc();
  }

  // Encode some access bounds checks.
  if (is_array_type(value)) {
    bounds_check(value, final_offset, type, tmp_guard);
  } else if (is_code_type(value) || is_code_type(type)) {
    check_code_access(value, final_offset, type, tmp_guard, mode);
  } else {
    check_data_obj_access(value, final_offset, type, tmp_guard);
  }

  build_reference_rec(value, final_offset, type, tmp_guard, mode, o.alignment,
                      scalar_step_list);

  return value;
}

/************************** Rereference building code *************************/

void
dereferencet::build_reference_rec(expr2tc &value, const expr2tc &offset,
                    const type2tc &type, const guardt &guard,
                    modet mode, unsigned long alignment,
                    std::list<expr2tc> *scalar_step_list)
{
  bool is_const_offs = is_constant_int2t(offset);

  // All accesses to code need no further construction
  if (is_code_type(value) || is_code_type(type)) {
    return;
  }

  // Specialised cases: struct refs to which we apply scalar steps, and
  // attempting to treat a byte array as a struct.
  if (is_constant_expr(offset)) {
    if (scalar_step_list && scalar_step_list->size() != 0) {
      // Base must be struct or array. However we're going to burst into flames
      // if we access a byte array as a struct; except that's legitimate when
      // we've just malloc'd it. So, special case that too.
      const type2tc &base_type_of_steps =
        (*scalar_step_list->front()->get_sub_expr(0))->type;

      if (is_array_type(value->type) &&
          to_array_type(value->type).subtype->get_width() == 8 &&
          (!is_array_type(base_type_of_steps) ||
           !to_array_type(base_type_of_steps).subtype->get_width() != 8)) {
        // Right, we're going to be accessing a byte array as not-a-byte-array.
        // Switch this access together.
        expr2tc offset_the_third =
          compute_pointer_offset(scalar_step_list->back());
#if 0
        dereference_callback.rename(offset_the_third);
#endif

        add2tc add2(offset->type, offset, offset_the_third);
        expr2tc new_offset = add2->simplify();
        if (is_nil_expr(new_offset))
          new_offset = add2;

        stitch_together_from_byte_array(value, type, new_offset);
      } else {
        construct_struct_ref_from_const_offset(value, offset,
                                               base_type_of_steps, guard);
        wrap_in_scalar_step_list(value, scalar_step_list, guard);
      }

      return;
    }
  }

  // All struct references to be built should be filtered out immediately
  if (is_structure_type(type)) {
    if (is_const_offs) {
      construct_struct_ref_from_const_offset(value, offset, type, guard);
    } else {
      construct_struct_ref_from_dyn_offset(value, offset, type, guard,
                                           scalar_step_list);
      if (scalar_step_list && scalar_step_list->size() != 0)
        wrap_in_scalar_step_list(value, scalar_step_list, guard);
    }
    return;
  }

  if (is_struct_type(value)) {
    assert(!is_struct_type(type));
    if (is_const_offs) {
      construct_from_const_struct_offset(value, offset, type, guard, mode);
    } else {
      construct_from_dyn_struct_offset(value, offset, type, guard, alignment,
                                       mode);
    }
    return;
  }

  if (is_union_type(value)) {
    // Huuurrrr. Just perform an access to the first element thing.
    const union_type2t &uni_type = to_union_type(value->type);
    assert(uni_type.members.size() != 0);
    value = member2tc(uni_type.members[0], value, uni_type.member_names[0]);

    build_reference_rec(value, offset, type, guard, mode, alignment,
                        scalar_step_list);
    return;
  }

  if (is_array_type(value) || is_string_type(value)) {
    construct_from_array(value, offset, type, guard, mode, alignment);
    return;
  }

  if (is_const_offs) {
    construct_from_const_offset(value, offset, type);
  } else {
    construct_from_dyn_offset(value, offset, type);
  }
}

void
dereferencet::construct_from_array(expr2tc &value, const expr2tc &offset,
                                   const type2tc &type, const guardt &guard,
                                   modet mode, unsigned long alignment)
{
  assert(is_array_type(value) || is_string_type(value));
  bool is_const_offset = is_constant_int2t(offset);

  const array_type2t arr_type = get_arr_type(value);
  type2tc arr_subtype = arr_type.subtype;

  unsigned long subtype_size = type_byte_size(*arr_subtype).to_ulong();
  unsigned long deref_size = type->get_width() / 8;

  if (is_array_type(arr_type.subtype)) {
    construct_from_multidir_array(value, offset, type, guard, alignment, mode);
    return;
  }

  constant_int2tc subtype_sz_expr(index_type2(), BigInt(subtype_size));
  div2tc div(index_type2(), offset, subtype_sz_expr);
  modulus2tc mod(index_type2(), offset, subtype_sz_expr);
  expr2tc div2 = div->simplify();
  expr2tc mod2 = mod->simplify();
  if (is_nil_expr(div2))
    div2 = div;
  if (is_nil_expr(mod2))
    mod2 = mod;

  if (is_structure_type(arr_subtype)) {
    value = index2tc(arr_subtype, value, div2);
    build_reference_rec(value, mod2, type, guard, mode, alignment);
    return;
  }

  assert(is_scalar_type(arr_subtype));

  // Two different ways we can access elements
  //  1) Just treat them as an element and select them out, possibly with some
  //     byte extracts applied to it
  //  2) Stitch everything together with extracts and concats.

  // Can we just select this out?
  if ((is_const_offset && deref_size <= subtype_size) ||
      (!is_const_offset && alignment >= subtype_size && deref_size <= subtype_size)) {
    // We're fine for just indexing and applying appropriate casts/extracts.
    // And here it is:
    value = index2tc(arr_subtype, value, div2);

    // Now assert that the appropriate alignment was used. There must be some
    // much more efficient way of doing this.
    // XXX short circuit byte accesses. Also, alignment might already guarentee
    // this.
    expr2tc mask_expr = gen_uint(deref_size - 1);
    bitand2tc anded(mask_expr->type, mask_expr, mod2);
    notequal2tc neq(anded, zero_uint);

    guardt tmp_guard = guard;
    tmp_guard.add(neq);
    alignment_failure("Incorrect alignment when accessing array element",
                      tmp_guard);

    // Finally, coerce the element to the final type. We may need to typecast
    // it; we might also need to byte extract it.
    if (deref_size == subtype_size) {
      // Just need to cast -- XXX this might break with endianness concerns.
      // If the condition here holds, it doesn't matter whether or not we're
      // const or dynamic offset.
      if (type != arr_subtype)
        value = typecast2tc(type, value);
    } else {
      // Badness has occurred; byte extract is needed.
      // XXX -- this should actually extract and stitch.
      value = byte_extract2tc(get_uint_type(deref_size * 8), value, mod2,
                              is_big_endian);
    }
  } else {
    // This either isn't aligned or is the wrong size. That might be fine if
    // further alignment rules are observed, so perform relevant assertions
    // and then stitch together from byte extracts.

    expr2tc mask_expr = gen_uint(deref_size - 1);
    bitand2tc anded(mask_expr->type, mask_expr, mod2);
    notequal2tc neq(anded, zero_uint);

    guardt tmp_guard = guard;
    tmp_guard.add(neq);
    alignment_failure("Incorrect alignment when accessing array element",
                      tmp_guard);

    // This will construct from whatever the subtype is...
    stitch_together_from_byte_array(value, type, mod2);
  }

  return;
}

void
dereferencet::construct_from_const_offset(expr2tc &value, const expr2tc &offset,
                                          const type2tc &type)
{

  const constant_int2t &theint = to_constant_int2t(offset);
  const type2tc &bytetype = get_uint8_type();

  assert(is_scalar_type(value));
  // We're accessing some kind of scalar type; might be a valid, correct
  // access, or we might need to be byte extracting it.

  if (theint.constant_value == 0 &&
      value->type->get_width() == type->get_width()) {
    // Offset is zero, and we select the entire contents of the field. We may
    // need to perform a cast though.
    if (!base_type_eq(value->type, type, ns)) {
      value = typecast2tc(type, value);
    }
  } else {
    // Either nonzero offset, or a smaller / bigger read.
    // XXX -- refactor to become concat based.
    value = byte_extract2tc(bytetype, value, offset, is_big_endian);
    if (type->get_width() != 8)
      value = typecast2tc(type, value);
  }
}

void
dereferencet::construct_from_const_struct_offset(expr2tc &value,
                        const expr2tc &offset, const type2tc &type,
                        const guardt &guard, modet mode)
{
  assert(is_struct_type(value->type));
  const struct_type2t &struct_type = to_struct_type(value->type);
  const mp_integer int_offset = to_constant_int2t(offset).constant_value;
  mp_integer access_size = type_byte_size(*type.get());

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    mp_integer m_offs = member_offset(struct_type, struct_type.member_names[i]);
    mp_integer m_size  = type_byte_size(*it->get());

    if (int_offset < m_offs) {
      // The offset is behind this field, but wasn't accepted by the previous
      // member. That means that the offset falls in the undefined gap in the
      // middled. Which might be an error -- reading from it definitely is,
      // but we might write to it in the course of memset.
      value = expr2tc();
      if (mode == WRITE) {
        // This write goes to an invalid symbol, but no assertion is encoded,
        // so it's entirely safe.
      } else {
        assert(mode == READ);
        // Oh dear. Encode a failure assertion.
        dereference_failure("pointer dereference",
                            "Dereference reads between struct fields", guard);
      }
    } else if (int_offset == m_offs) {
      // Does this over-read?
      if (access_size > m_size) {
        dereference_failure("pointer dereference",
                            "Over-sized read of struct field", guard);
        value = expr2tc();
        return;
      }

      // XXX -- what about under-reads?

      // If it's at the start of a field, there's no need for further alignment
      // concern.
      expr2tc res = member2tc(*it, value, struct_type.member_names[i]);

      if (!is_scalar_type(*it)) {
        // We have to do even more extraction...
        build_reference_rec(res, zero_uint, type, guard, mode);
      }

      value = res;
      return;
    } else if (int_offset > m_offs &&
              (int_offset - m_offs + access_size <= m_size)) {
      // This access is in the bounds of this member, but isn't at the start.
      // XXX that might be an alignment error.
      // In the meantime, byte extract.
      expr2tc memb = member2tc(*it, value, struct_type.member_names[i]);
      constant_int2tc new_offs(index_type2(), int_offset - m_offs);

      // Extract.
      build_reference_rec(memb, new_offs, type, guard, mode);
      value = memb;
      return;
    } else if (int_offset < (m_offs + m_size)) {
      // This access starts in this field, but by process of elimination,
      // doesn't end in it. Which means reading padding data (or an alignment
      // error), which are both bad.
      alignment_failure("Misaligned access to struct field", guard);
      value = expr2tc();
      return;
    }

    // Wasn't that field.
    i++;
  }

  // Fell out of that struct -- means we've accessed out of bounds. Code at
  // a higher level will encode an assertion to this effect.
  value = expr2tc();
}

void
dereferencet::construct_from_dyn_struct_offset(expr2tc &value,
                                  const expr2tc &offset, const type2tc &type,
                                  const guardt &guard, unsigned long alignment,
                                  modet mode,
                                  const expr2tc *failed_symbol)
{
  // For each element of the struct, look at the alignment, and produce an
  // appropriate access (that we'll switch on).
  assert(is_struct_type(value->type));
  const struct_type2t &struct_type = to_struct_type(value->type);
  unsigned int access_sz = type->get_width() / 8;

  expr2tc failed_container;
  if (failed_symbol == NULL)
    failed_container = make_failed_symbol(type);
  else
    failed_container = *failed_symbol;

  // A list of guards, and outcomes. The result should be a gigantic
  // if-then-else chain based on those guards.
  std::list<std::pair<expr2tc, expr2tc> > extract_list;

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    mp_integer offs = member_offset(struct_type, struct_type.member_names[i]);

    // Compute some kind of guard
    unsigned int field_size = (*it)->get_width() / 8;
    // Round up to word size
    unsigned int word_mask = (config.ansi_c.word_size / 8) - 1;
    field_size = (field_size + word_mask) & (~word_mask);
    expr2tc field_offs = gen_uint(offs.to_ulong());
    expr2tc field_top = gen_uint(offs.to_ulong() + field_size);
    expr2tc lower_bound = greaterthanequal2tc(offset, field_offs);
    expr2tc upper_bound = lessthan2tc(offset, field_top);
    expr2tc field_guard = and2tc(lower_bound, upper_bound);

    if (is_struct_type(*it)) {
      // Handle recursive structs
      expr2tc new_offset = sub2tc(offset->type, offset, field_offs);
      expr2tc field = member2tc(*it, value, struct_type.member_names[i]);
      construct_from_dyn_struct_offset(field, new_offset, type, guard,
                                       alignment, mode, &failed_container);
      extract_list.push_back(std::pair<expr2tc,expr2tc>(field_guard, field));
    } else if (is_union_type(*it)) {
      // Take the union, take the first field, and consider a dynamiclly offset
      // assignment into the first element. This is a massive approximation;
      // what we really need is a well-reasoned representation of unions. Like
      // the byte model, say
      expr2tc new_offset = sub2tc(offset->type, offset, field_offs);
      expr2tc field = member2tc(*it, value, struct_type.member_names[i]);

      const union_type2t &uni_type = to_union_type(field->type);
      assert(uni_type.members.size() != 0);
      field = member2tc(uni_type.members[0], field, uni_type.member_names[0]);
      if (is_struct_type(field)) {
        construct_from_dyn_struct_offset(field, new_offset, type, guard,
                                         alignment, mode, &failed_container);
      } else {
        build_reference_rec(field, new_offset, type, guard, mode, alignment);
      }

      extract_list.push_back(std::pair<expr2tc,expr2tc>(field_guard, field));
    } else if (is_array_type(*it)) {
      expr2tc new_offset = sub2tc(offset->type, offset, field_offs);
      expr2tc field = member2tc(*it, value, struct_type.member_names[i]);
      build_reference_rec(field, new_offset, type, guard, mode, alignment);
      extract_list.push_back(std::pair<expr2tc,expr2tc>(field_guard, field));
    } else if (access_sz > ((*it)->get_width() / 8)) {
      guardt newguard(guard);
      newguard.add(field_guard);
      dereference_failure("pointer dereference",
                          "Oversized field offset", guard);
      // Push nothing back, allow fall-through of the if-then-else chain to
      // resolve to a failed deref symbol.
    } else if (alignment >= (config.ansi_c.word_size / 8)) {
      // This is fully aligned, just pull it out and possibly cast,
      // XXX endian?
      expr2tc field = member2tc(*it, value, struct_type.member_names[i]);
      if (!base_type_eq(field->type, type, ns))
        field = typecast2tc(type, field);
      extract_list.push_back(std::pair<expr2tc,expr2tc>(field_guard, field));
    } else {
      // Not fully aligned; devolve to byte extract. There may be ways to
      // optimise this further, but that's quite meh right now.
      // XXX -- stitch together with concats?
      expr2tc new_offset = sub2tc(offset->type, offset, field_offs);
      expr2tc field = member2tc(*it, value, struct_type.member_names[i]);
      field = byte_extract2tc(get_uint8_type(), field, new_offset,
                              is_big_endian);
      if (type->get_width() != 8)
        field = typecast2tc(type, field);

      extract_list.push_back(std::pair<expr2tc,expr2tc>(field_guard, field));
    }

    i++;
  }

  // Build up the new value, switching on the field guard, with the failed
  // symbol at the base.
  expr2tc new_value = failed_container;
  for (std::list<std::pair<expr2tc, expr2tc> >::const_iterator
       it = extract_list.begin(); it != extract_list.end(); it++) {
    new_value = if2tc(type, it->first, it->second, new_value);
  }

  value = new_value;
}

void
dereferencet::construct_from_dyn_offset(expr2tc &value, const expr2tc &offset,
                                const type2tc &type)
{
  expr2tc orig_value = value;

  // Else, in the case of a scalar access at the bottom,
  assert(config.ansi_c.endianess != configt::ansi_ct::NO_ENDIANESS);
  assert(is_scalar_type(value));

  // Ensure we're dealing with a BV.
  if (!is_number_type(value->type)) {
    value = typecast2tc(get_uint_type(value->type->get_width()), value);
  }

  const type2tc &bytetype = get_uint8_type();
  value = byte_extract2tc(bytetype, value, offset, is_big_endian);

  // XXX jmorse - temporary, while byte extract is still covered in bees.
  value = typecast2tc(type, value);
}

void
dereferencet::construct_from_multidir_array(expr2tc &value,
                              const expr2tc &offset,
                              const type2tc &type, const guardt &guard,
                              unsigned long alignment,
                              modet mode)
{
  assert(is_array_type(value) || is_string_type(value));
  const array_type2t arr_type = get_arr_type(value);

  // Right: any access across the boundry of the outer dimension of this array
  // is an alignment violation, I think. (It isn't for byte arrays, worry about
  // that later XXX).
  // So, divide the offset by size of the inner dimention, make an index2t, and
  // construct a reference to that.
  mp_integer subtype_sz = type_byte_size(*arr_type.subtype);
  constant_int2tc subtype_sz_expr(index_type2(), subtype_sz);
  div2tc div(index_type2(), offset, subtype_sz_expr);
  modulus2tc mod(index_type2(), offset, subtype_sz_expr);

  expr2tc idx = div->simplify();
  if (is_nil_expr(idx))
    idx = div;

  index2tc outer_idx(arr_type.subtype, value, idx);
  value = outer_idx;

  idx = mod->simplify();
  if (is_nil_expr(idx))
    idx = mod;

  build_reference_rec(value, idx, type, guard, mode, alignment);
}

void
dereferencet::construct_struct_ref_from_const_offset(expr2tc &value,
             const expr2tc &offs, const type2tc &type, const guardt &guard)
{
  // Minimal effort: the moment that we can throw this object out due to an
  // incompatible type, we do.
  const constant_int2t &intref = to_constant_int2t(offs);

  if (is_array_type(value->type)) {
    const array_type2t &arr_type = to_array_type(value->type);

    if (!is_struct_type(arr_type.subtype) && !is_array_type(arr_type.subtype)) {
      // Can't handle accesses to anything else.
      dereference_failure("Memory model",
                          "Object accessed with incompatible base type", guard);
      return;
    }

    // Crazyness: we might be returning a reference to an array, not a struct,
    // because this method needs renaming.
    if (is_array_type(type)) {
      const array_type2t target_type = to_array_type(type);
      // If subtype sizes match, then we're as good as we're going to be for
      // returning a reference to the desired subarray.
      if (target_type.subtype->get_width() == arr_type.subtype->get_width())
        return;
    }

    // Create an access to an array index. Alignment will be handled at a lower
    // layer, because we might not be able to detect that it's valid (structs
    // within structs).
    mp_integer subtype_size = type_byte_size(*arr_type.subtype.get());
    mp_integer idx = intref.constant_value / subtype_size;
    mp_integer mod = intref.constant_value % subtype_size;

    expr2tc idx_expr = gen_uint(idx.to_ulong());
    expr2tc mod_expr = gen_uint(mod.to_ulong());

    value = index2tc(arr_type.subtype, value, idx_expr);

    construct_struct_ref_from_const_offset(value, mod_expr, type, guard);
  } else if (is_struct_type(value->type)) {
    // Right. In this situation, there are several possibilities. First, if the
    // offset is zero, and the struct type is compatible, we've succeeded.
    // If the offset isn't zero, then there are some possibilities:
    //
    //   a) it's a misaligned access, which is an error
    //   b) there's a struct within a struct here that we should recurse into.

    if (intref.constant_value == 0) {
      // Success?
      if (dereference_type_compare(value, type)) {
        // Good, just return this expression. Uh. Yeah, that is all.
        return;
      }
    }

    // If it's not compatible, recurse into the next relevant field to see if
    // we can construct a ref in there. This would match structs within structs
    // (but compatible check already gets that;), arrays of structs; and other
    // crazy inside structs.

    const struct_type2t &struct_type = to_struct_type(value->type);
    unsigned int i = 0;
    forall_types(it, struct_type.members) {
      mp_integer offs = member_offset(struct_type,
                                      struct_type.member_names[i]);
      mp_integer size = type_byte_size(*(*it).get());

      if (!is_scalar_type(*it) &&
            intref.constant_value >= offs &&
            intref.constant_value <= (offs + size)) {
        // It's this field. Don't make a decision about whether it's correct
        // or not, recurse to make that happen.
        mp_integer new_offs = intref.constant_value - offs;
        expr2tc offs_expr = gen_uint(new_offs.to_ulong());
        value = member2tc(*it, value, struct_type.member_names[i]);
        construct_struct_ref_from_const_offset(value, offs_expr, type, guard);
        return;
      }
      i++;
    }

    // Fell out of that loop. Either this offset is out of range, or lies in
    // padding.
    dereference_failure("Memory model", "Object accessed with illegal offset",
                        guard);
  } else if (is_union_type(value)) {
    // XXX -- this only deals with a very shallow level of unioning.

    if (base_type_eq(value->type, type, ns))
      return;

    const union_type2t &uni = to_union_type(value->type);
    unsigned int i = 0;
    forall_types(it, uni.members) {
      if (base_type_eq(*it, type, ns)) {
        // We have a subtype that matches the type we want to be getting.
        member2tc memb(*it, value, uni.member_names[i]);
        value = memb;
        return;
      }
      i++;
    }
    dereference_failure("Memory model",
                        "Object accessed with incompatible base type",
                        guard);
  } else {
    dereference_failure("Memory model",
                        "Object accessed with incompatible base type", guard);
  }

  return;
}

void
dereferencet::construct_struct_ref_from_dyn_offset(expr2tc &value,
             const expr2tc &offs, const type2tc &type, const guardt &guard,
             std::list<expr2tc> *scalar_step_list __attribute__((unused)))
{
  // This is much more complicated -- because we don't know the offset here,
  // we need to go through all the possible fields that this might (legally)
  // resolve to and switch on them; then assert that one of them is accessed.
  // So:
  std::list<std::pair<expr2tc, expr2tc> > resolved_list;

  construct_struct_ref_from_dyn_offs_rec(value, offs, type, true_expr,
                                         resolved_list);

  if (resolved_list.size() == 0) {
    // No legal accesses.
    value = expr2tc();
    dereference_failure("Memory model",
                        "Object accessed with incompatible base type", guard);
    return;
  }

  // Switch on the available offsets.
  expr2tc result = make_failed_symbol(type);
  for (std::list<std::pair<expr2tc, expr2tc> >::const_iterator
       it = resolved_list.begin(); it != resolved_list.end(); it++) {
    result = if2tc(type, it->first, it->second, result);
  }

  value = result;

  // Finally, record an assertion that if none of those accesses were legal,
  // then it's an illegal access.
  expr2tc accuml = false_expr;
  for (std::list<std::pair<expr2tc, expr2tc> >::const_iterator
       it = resolved_list.begin(); it != resolved_list.end(); it++) {
    accuml = or2tc(accuml, it->first);
  }

  accuml = not2tc(accuml); // Creates a new 'not' expr. Doesn't copy construct.
  guardt tmp_guard = guard;
  tmp_guard.add(accuml);
  dereference_failure("Memory model",
                      "Object accessed with incompatible base type", tmp_guard);
}

void
dereferencet::construct_struct_ref_from_dyn_offs_rec(const expr2tc &value,
                              const expr2tc &offs, const type2tc &type,
                              const expr2tc &accuml_guard,
                              std::list<std::pair<expr2tc, expr2tc> > &output)
{
  // Look for all the possible offsets that could result in a legitimate access
  // to the given (struct?) type. Insert into the output list, with a guard
  // based on the 'offs' argument, that identifies when this field is legally
  // accessed.

  if (is_array_type(value->type)) {
    const array_type2t &arr_type = to_array_type(value->type);
    // We can legally access various offsets into arrays. Generate an index
    // and recurse. The complicate part is the new offset and guard: we need
    // to guard for offsets that are inside this array, and modulus the offset
    // by the array size.
    mp_integer subtype_size = type_byte_size(*arr_type.subtype.get());
    expr2tc sub_size = gen_uint(subtype_size.to_ulong());
    expr2tc div = div2tc(offs->type, offs, sub_size);
    expr2tc mod = modulus2tc(offs->type, offs, sub_size);
    expr2tc index = index2tc(arr_type.subtype, value, div);

    // We have our index; now compute guard/offset. Guard expression is
    // (offs >= 0 && offs < size_of_this_array)
    expr2tc new_offset = mod;
    expr2tc gte = greaterthanequal2tc(offs, zero_uint);
    expr2tc lt = lessthan2tc(offs, arr_type.array_size);
    expr2tc range_guard = and2tc(accuml_guard, and2tc(gte, lt));

    construct_struct_ref_from_dyn_offs_rec(index, new_offset, type, range_guard,
                                           output);
    return;
  } else if (is_struct_type(value->type)) {
    // OK. If this type is compatible and matches, we're good. There can't
    // be any subtypes in this struct that match because then it'd be defined
    // recursively (XXX -- is this true?).
    expr2tc tmp = value;
    if (dereference_type_compare(tmp, type)) {
      // Excellent. Guard that the offset is zero and finish.
      expr2tc offs_is_zero = and2tc(accuml_guard, equality2tc(offs, zero_uint));
      output.push_back(std::pair<expr2tc, expr2tc>(offs_is_zero, tmp));
      return;
    }

    // It's not compatible, but a subtype may be. Iterate over all of them.
    const struct_type2t &struct_type = to_struct_type(value->type);
    unsigned int i = 0;
    forall_types(it, struct_type.members) {
      // Quickly skip over scalar subtypes.
      if (is_scalar_type(*it))
        continue;

      mp_integer memb_offs = member_offset(struct_type,
                                      struct_type.member_names[i]);
      mp_integer size = type_byte_size(*(*it).get());
      expr2tc memb_offs_expr = gen_uint(memb_offs.to_ulong());
      expr2tc limit_expr = gen_uint(memb_offs.to_ulong() + size.to_ulong());
      expr2tc memb = member2tc(*it, value, struct_type.member_names[i]);

      // Compute a guard and update the offset for an access to this field.
      // Guard is that the offset is in the range of this field. Offset has
      // offset to this field subtracted.
      expr2tc new_offset = sub2tc(offs->type, offs, memb_offs_expr);
      expr2tc gte = greaterthanequal2tc(offs, memb_offs_expr);
      expr2tc lt = lessthan2tc(offs, limit_expr);
      expr2tc range_guard = and2tc(accuml_guard, and2tc(gte, lt));

      construct_struct_ref_from_dyn_offs_rec(memb, new_offset, type,
                                             range_guard, output);
    }
  } else {
    // Not legal
    return;
  }
}

/**************************** Dereference utilities ***************************/

void
dereferencet::dereference_failure(const std::string &error_class,
                                  const std::string &error_name,
                                  const guardt &guard)
{
  // This just wraps dereference failure in a no-pointer-check check.
  if(!options.get_bool_option("no-pointer-check")) {
    dereference_callback.dereference_failure( error_class, error_name, guard);
  }
}

void
dereferencet::alignment_failure(const std::string &error_name,
                                const guardt &guard)
{
  // This just wraps dereference failure in a no-pointer-check check.
  if(!options.get_bool_option("no-align-check")) {
    dereference_failure("Pointer alignment", error_name, guard);
  }
}

void
dereferencet::stitch_together_from_byte_array(expr2tc &value,
                                              const type2tc &type,
                                              const expr2tc &offset)
{
  const array_type2t &arr_type = to_array_type(value->type);
  // Unstructured array access. First, check alignment.
  unsigned int subtype_sz = arr_type.subtype->get_width() / 8;

  unsigned int target_bytes = type->get_width() / 8;
  expr2tc accuml;
  expr2tc accuml_offs = offset;
  type2tc subtype = arr_type.subtype;

  for (unsigned int i = 0; i < target_bytes; i += subtype_sz) {
    expr2tc elem = index2tc(subtype, value, accuml_offs);

    if (is_nil_expr(accuml)) {
      accuml = elem;
    } else {
      // XXX -- byte order.
      type2tc res_type =
        get_uint_type(accuml->type->get_width() + (subtype_sz * 8));
      accuml = concat2tc(res_type, accuml, elem);
    }

    accuml_offs = add2tc(offset->type, accuml_offs, one_uint);
  }

  // That's going to come out as a bitvector;
  if (type != accuml->type) {
    // XXX -- we might be selecting a char out of an int array, or something
    //        This really needs to consider the initial offset into these array
    //        elements. Use alignment and apply a byte extract?
    // XXX -- byte order.
    //assert(type->get_width() == accuml->type->get_width());
    accuml = typecast2tc(type, accuml);
  }

  value = accuml;
}

void dereferencet::valid_check(
  const expr2tc &object,
  const guardt &guard,
  modet mode)
{

  const expr2tc &symbol = get_symbol(object);

  if (is_constant_string2t(symbol))
  {
    // always valid, but can't write

    if(mode==WRITE)
    {
      dereference_failure("pointer dereference",
                          "write access to string constant", guard);
    }
  }
  else if (is_nil_expr(symbol))
  {
    // always "valid", shut up
    return;
  }
  else if (is_symbol2t(symbol))
  {
    // Hacks, but as dereferencet object isn't persistent, necessary. Fix by
    // making dereferencet persistent.
    if (has_prefix(to_symbol2t(symbol).thename.as_string(), "symex::invalid_object"))
      return;

    const symbolt &sym = ns.lookup(to_symbol2t(symbol).thename);
    if (has_prefix(sym.name.as_string(), "symex_dynamic::")) {
      // Assert thtat it hasn't (nondeterministically) been invalidated.
      address_of2tc addrof(symbol->type, symbol);
      valid_object2tc valid_expr(addrof);
      not2tc not_valid_expr(valid_expr);

      guardt tmp_guard(guard);
      tmp_guard.move(not_valid_expr);
      dereference_failure("pointer dereference", "invalidated dynamic object",
                          tmp_guard);
    } else {
      // Not dynamic; if we're in free mode, that's an error.
      if(mode==FREE)
      {
        dereference_failure("pointer dereference",
                            "free() of non-dynamic memory", guard);
        return;
      }
    }
  }
}

void dereferencet::bounds_check(const expr2tc &expr, const expr2tc &offset,
                                const type2tc &type, const guardt &guard)
{
  if(options.get_bool_option("no-bounds-check"))
    return;

  unsigned long access_size = type_byte_size(*type).to_ulong();

  assert(is_array_type(expr) || is_string_type(expr));

  expr2tc arrsize;
  const symbolt &sym = ns.lookup(to_symbol2t(expr).thename);
  if (has_prefix(sym.name.as_string(), "symex_dynamic::")) {
    // Construct a dynamic_size irep.
    address_of2tc addrof(expr->type, expr);
    arrsize = dynamic_size2tc(addrof);
  } else {
    // Calculate size from type.

    // Dance around getting the array type normalised.
    type2tc new_string_type;
    const array_type2t arr_type = get_arr_type(expr);

    // XXX -- arrays were assigned names, but we're skipping that for the moment
    // std::string name = array_name(ns, expr.source_value);

    // Firstly, bail if this is an infinite sized array. There are no bounds
    // checks to be performed.
    if (arr_type.size_is_infinite)
      return;

    // Secondly, try to calc the size of the array.
    unsigned long subtype_size_int
      = type_byte_size(*arr_type.subtype).to_ulong();
    constant_int2tc subtype_size(get_uint32_type(), BigInt(subtype_size_int));
    arrsize = mul2tc(get_uint32_type(), arr_type.array_size, subtype_size);
  }

  // Then, expressions as to whether the access is over or under the array
  // size.
  constant_int2tc access_size_e(get_uint32_type(), BigInt(access_size));
  add2tc upper_byte(get_uint32_type(), offset, access_size_e);

  greaterthan2tc gt(upper_byte, arrsize);

  // Report these as assertions; they'll be simplified away if they're constant

  guardt tmp_guard1(guard);
  tmp_guard1.move(gt);
  dereference_failure("array bounds", "array bounds violated", tmp_guard1);
}

void
dereferencet::wrap_in_scalar_step_list(expr2tc &value,
                                       std::list<expr2tc> *scalar_step_list,
                                       const guardt &guard)
{
  // Check that either the base type that these steps are applied to matches
  // the type of the object we're wrapping in these steps. It's a type error
  // if there isn't a match.
  expr2tc base_of_steps = *scalar_step_list->front()->get_sub_expr(0);
  if (dereference_type_compare(value, base_of_steps->type)) {
    // We can just reconstruct this.
    expr2tc accuml = value;
    for (std::list<expr2tc>::const_iterator it = scalar_step_list->begin();
         it != scalar_step_list->end(); it++) {
      expr2tc tmp = *it;
      *tmp.get()->get_sub_expr_nc(0) = accuml;
      accuml = tmp;
    }
    value = accuml;
  } else {
    // We can't reconstruct this. Go crazy instead.
    // XXX -- there's a line in the C spec, appendix G or whatever, saying that
    // accessing an object with an (incompatible) type other than its base type
    // is undefined behaviour. Should totally put that in the error message.
    dereference_failure("Memory model",
                        "Object accessed with incompatible base type", guard);
    value = expr2tc();
  }
}

void
dereferencet::check_code_access(expr2tc &value, const expr2tc &offset,
                                const type2tc &type, const guardt &guard,
                                modet mode)
{
  if (is_code_type(value) && !is_code_type(type)) {
    dereference_failure("Code separation", "Program code accessed with non-code"
                        " type", guard);
  } else if (!is_code_type(value) && is_code_type(type)) {
    dereference_failure("Code separation", "Data object accessed with code "
                        "type", guard);
  } else {
    assert(is_code_type(value) && is_code_type(type));

    if (mode != READ) {
      dereference_failure("Code separation", "Program code accessed in write or"
                          " free mode", guard);
    }

    // Only other constraint is that the offset has to be zero; there are no
    // other rules about what code objects look like.
    notequal2tc neq(offset, zero_uint);
    guardt tmp_guard = guard;
    tmp_guard.add(neq);
    dereference_failure("Code separation", "Program code accessed with non-zero"
                        " offset", tmp_guard);
  }

  // As for setting the 'value', it's currently already set to the base code
  // object. There's nothing we can actually change it to to mean anything, so
  // don't fiddle with it.
  return;
}

void
dereferencet::check_data_obj_access(const expr2tc &value, const expr2tc &offset,
                                    const type2tc &type, const guardt &guard)
{
  assert(!is_array_type(value));

  unsigned long data_sz = type_byte_size(*value->type).to_ulong();
  unsigned long access_sz = type_byte_size(*type).to_ulong();
  expr2tc data_sz_e = gen_uint(data_sz);
  expr2tc access_sz_e = gen_uint(access_sz);

  // Only erronous thing we check for right now is that the offset is out of
  // bounds, misaligned access happense elsewhere. The highest byte read is at
  // offset+access_sz-1, so check fail if the (offset+access_sz) > data_sz.
  // Lower bound not checked, instead we just treat everything as unsigned,
  // which has the same effect.
  add2tc add(access_sz_e->type, offset, access_sz_e);
  greaterthan2tc gt(add, data_sz_e);

  guardt tmp_guard = guard;
  tmp_guard.add(gt);
  dereference_failure("pointer dereference",
                      "Access to object out of bounds", tmp_guard);

  // Also, if if it's a scalar, check that the access being made is aligned.
  if (is_scalar_type(type)) {
    expr2tc mask_expr = gen_uint(access_sz - 1);
    bitand2tc anded(mask_expr->type, mask_expr, offset);
    notequal2tc neq(anded, zero_uint);

    guardt tmp_guard2 = guard;
    tmp_guard2.add(neq);
    alignment_failure("Incorrect alignment when accessing data object",
                      tmp_guard2);
  }
}
