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

// Look for the base of an expression such as &a->b[1];, where all we're doing
// is performing some pointer arithmetic, rather than actually performing some
// dereference operation.
static inline expr2tc get_base_dereference(const expr2tc &e)
{

  // XXX -- do we need to consider if2t's? And how?
  if (is_member2t(e)) {
    return get_base_dereference(to_member2t(e).source_value);
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

void
dereferencet::dereference_expr(
  expr2tc &expr,
  guardt &guard,
  const modet mode,
  bool checks_only)
{

  if (!has_dereference(expr))
    return;

  // Preliminary, guard munging tests. 
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
  else if (is_if2t(expr))
  {
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
      dereference_expr(ifref.true_value, guard, mode, checks_only);
      guard.resize(old_guard);
    }

    if (o2) {
      unsigned old_guard=guard.size();
      not2tc tmp(ifref.cond);
      guard.move(tmp);
      dereference_expr(ifref.false_value, guard, mode, checks_only);
      guard.resize(old_guard);
    }

    return;
  }

  // Crazy combinations of & and * that don't actually lead to a deref:
  if (is_address_of2t(expr))
  {
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
      return;
    } else {
      // This might, alternately, be a chain of member and indexes applied to
      // a dereference. In which case what we're actually doing is computing
      // some pointer arith, manually.
      expr2tc base = get_base_dereference(addrof.ptr_obj);
      if (!is_nil_expr(base)) {
        //  We have a base. There may be additional dereferences in it.
        dereference_expr(base, guard, mode, checks_only);
        // Now compute the pointer offset involved.
        expr2tc offs = compute_pointer_offset(addrof.ptr_obj);
        assert(!is_nil_expr(offs) && "Pointer offset of index/member "
               "combination should be valid int");

        // Cast to a byte pointer; add; cast back. Can't think of a better way
        // to produce safe pointer arithmetic right now.
        expr2tc output =
          typecast2tc(type2tc(new pointer_type2t(get_uint8_type())), base);
        output = add2tc(output->type, output, offs);
        output = typecast2tc(base->type, output);
        expr = output;
        return;
      }
    }
  }

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

    if ((is_scalar_type(expr) || is_code_type(expr) || checks_only)) {
    //assert((is_scalar_type(expr) || is_code_type(expr) || checks_only)
    //   && "Can't dereference to a nonscalar type");
      expr2tc tmp_obj = deref.value;
      expr2tc result = dereference(tmp_obj, deref.type, guard, mode,
                                   &scalar_step_list);
      expr = result;
    } else {
      // Nonscalar dereference; pretend we know what we're doing for a moment
      expr2tc tmp_obj = deref.value;
      expr2tc result = dereference(tmp_obj, deref.type, guard, mode,
                                   &scalar_step_list);
      // And now, remove any intermediate indexes or members.
      while (!dereference_type_compare(result, expr->type)) {
        if (is_index2t(result)) {
          result = to_index2t(result).source_value;
        } else if (is_member2t(result)) {
          result = to_member2t(result).source_value;
        } else {
          std::cerr << "Erronous struct dereference" << std::endl;
          abort();
        }
      }
      expr = result;
    }
  } else if (is_index2t(expr) &&
             is_pointer_type(to_index2t(expr).source_value)) {
    std::list<expr2tc> scalar_step_list;
    assert((is_scalar_type(expr) || is_code_type(expr) || checks_only)
           && "Can't dereference to a nonscalar type");
    index2t &idx = to_index2t(expr);

    // first make sure there are no dereferences in there
    dereference_expr(idx.index, guard, dereferencet::READ);
    dereference_expr(idx.source_value, guard, mode);

    add2tc tmp(idx.source_value->type, idx.source_value, idx.index);
    // Result discarded.
    expr = dereference(tmp, tmp->type, guard, mode, &scalar_step_list);
  } else if (is_non_scalar_expr(expr)) {
    // The result of this expression should be scalar: we're transitioning
    // from a scalar result to a nonscalar result.
    // Unless we're doing something crazy with multidimensional arrays and
    // address_of, for example, where no dereference is involved. In that case,
    // bail.
    bool contains_deref = has_dereference(expr);
    if (!contains_deref)
      return;

//    assert(is_scalar_type(expr));

    std::list<expr2tc> scalar_step_list;
    expr2tc res = dereference_expr_nonscalar(expr, guard, mode,
                                             scalar_step_list, checks_only);
    assert(scalar_step_list.size() == 0); // Should finish empty.

    // If a dereference successfully occurred, replace expr at this level.
    // XXX -- explain this better.
    if (!is_nil_expr(res))
      expr = res;
  } else {
    Forall_operands2(it, idx, expr) {
      if (is_nil_expr(*it))
        continue;

      dereference_expr(*it, guard, mode, checks_only);
    }
  }
}

expr2tc
dereferencet::dereference_expr_nonscalar(
  expr2tc &expr,
  guardt &guard,
  const modet mode,
  std::list<expr2tc> &scalar_step_list,
  bool checks_only)
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
    dereference_expr(index.source_value, guard, mode);
    dereference_expr(index.index, guard, mode);

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
                                        mode, scalar_step_list, checks_only);
      scalar_step_list.pop_front();
    } else if (is_index2t(expr)) {
      dereference_expr(to_index2t(expr).index, guard, mode);
      scalar_step_list.push_front(expr);
      res = dereference_expr_nonscalar(to_index2t(expr).source_value, guard,
                                       mode, scalar_step_list, checks_only);
      scalar_step_list.pop_front();
    } else if (is_if2t(expr)) {
      // XXX - make this work similarly to dereference_expr.
      guardt g1 = guard, g2 = guard;
      if2t &theif = to_if2t(expr);
      g1.add(theif.cond);
      g2.add(not2tc(theif.cond));

      scalar_step_list.push_front(theif.true_value);
      expr2tc res1 = dereference_expr_nonscalar(theif.true_value, g1, mode,
                                                scalar_step_list, checks_only);
      scalar_step_list.pop_front();

      scalar_step_list.push_front(theif.false_value);
      expr2tc res2 = dereference_expr_nonscalar(theif.false_value, g2, mode,
                                                scalar_step_list, checks_only);
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
                                      scalar_step_list, checks_only);
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

expr2tc
dereferencet::dereference(
  const expr2tc &src,
  const type2tc &to_type,
  const guardt &guard,
  const modet mode,
  std::list<expr2tc> *scalar_step_list)
{
  expr2tc dest = src;
  assert(is_pointer_type(dest));

  // Target type is either a scalar type passed down to us, or we have a chain
  // of scalar steps available that end up at a scalar type. The result of this
  // dereference should be a scalar, via whatever means.
  type2tc type = (!is_nil_type(to_type))
    ? to_type : scalar_step_list->back()->type;

  // save the dest for later, dest might be destroyed
  const expr2tc deref_expr(dest);

  // collect objects dest may point to
  value_setst::valuest points_to_set;

  dereference_callback.get_value_set(dest, points_to_set);

  // now build big case split
  // only "good" objects

  expr2tc value;

  for(value_setst::valuest::const_iterator
      it=points_to_set.begin();
      it!=points_to_set.end();
      it++)
  {
    expr2tc new_value, pointer_guard;

    build_reference_to(*it, mode, dest, type, new_value, pointer_guard, guard,
                       scalar_step_list);

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

  if (is_nil_expr(value))
  {
    // first see if we have a "failed object" for this pointer

    const symbolt *failed_symbol;

    if (dereference_callback.has_failed_symbol(deref_expr, failed_symbol))
    {
      // yes!
      exprt tmp_val = symbol_expr(*failed_symbol);
      migrate_expr(tmp_val, value);

      // Wrap it in the scalar step list, to ensure it has the right type.
      if (scalar_step_list->size() != 0)
        wrap_in_scalar_step_list(value, scalar_step_list, guard);

    }
    else
    {
      value = make_failed_symbol(type);
    }
  }

  dest = value;
  return dest;
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

  // really different

  return false;
}

void dereferencet::build_reference_to(
  const expr2tc &what,
  const modet mode,
  const expr2tc &deref_expr,
  const type2tc &type,
  expr2tc &value,
  expr2tc &pointer_guard,
  const guardt &guard,
  std::list<expr2tc> *scalar_step_list)
{
  value = expr2tc();
  pointer_guard = false_expr;

  if (is_unknown2t(what) || is_invalid2t(what))
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      // constraint that it actually is an invalid pointer

      invalid_pointer2tc invalid_pointer_expr(deref_expr);

      // produce new guard

      guardt tmp_guard(guard);
      tmp_guard.move(invalid_pointer_expr);
      dereference_callback.dereference_failure(
        "pointer dereference",
        "invalid pointer",
        tmp_guard);
    }

    return;
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
    if(!options.get_bool_option("no-pointer-check"))
    {
      type2tc nullptrtype = type2tc(new pointer_type2t(type));
      symbol2tc null_ptr(nullptrtype, "NULL");

      same_object2tc pointer_guard(deref_expr, null_ptr);

      guardt tmp_guard(guard);
      tmp_guard.add(pointer_guard);

      dereference_callback.dereference_failure(
        "pointer dereference",
        "NULL pointer", tmp_guard);
    }

    // Don't build a reference to this. You can't actually access NULL, and the
    // solver will only get confused.
    return;
  }
  else
  {
    value = object;

    type2tc ptr_type = type2tc(new pointer_type2t(object->type));
    address_of2tc obj_ptr(ptr_type, object);

    pointer_guard = same_object2tc(deref_expr, obj_ptr);

    guardt tmp_guard(guard);
    tmp_guard.add(pointer_guard);

    valid_check(object, tmp_guard, mode);

    expr2tc orig_value = value;
    value = get_base_object(value);
    expr2tc additional_offset = compute_pointer_offset(orig_value);
    add2tc add(o.offset->type, o.offset, additional_offset);
    expr2tc final_offset = add->simplify();
    if (is_nil_expr(final_offset))
      final_offset = add;

    if (is_constant_expr(final_offset)) {
      // Hurrrrrr
      if (is_struct_type(type)) {
        construct_struct_ref_from_const_offset(value, final_offset, type,
                                               tmp_guard, scalar_step_list);
        if (scalar_step_list->size() != 0)
          wrap_in_scalar_step_list(value, scalar_step_list, guard);
      } else {
        const constant_int2t &theint = to_constant_int2t(final_offset);
        if (theint.constant_value.to_ulong() == 0)
          construct_from_zero_offset(value, type, tmp_guard, scalar_step_list);
        else
          construct_from_const_offset(value, final_offset, type, tmp_guard,
                                      scalar_step_list);
      }
    } else {
      expr2tc offset = pointer_offset2tc(index_type2(), deref_expr);
      if (is_struct_type(type)) {
        construct_struct_ref_from_dyn_offset(value, offset, type,
                                             tmp_guard, scalar_step_list);
        if (scalar_step_list->size() != 0)
          wrap_in_scalar_step_list(value, scalar_step_list, guard);
      } else {
        construct_from_dyn_offset(value, offset, type, tmp_guard, o.alignment);
      }
    }
  }
}

void
dereferencet::construct_from_zero_offset(expr2tc &value, const type2tc &type,
                                          const guardt &guard,
                                          std::list<expr2tc> *scalar_step_list)
{

  expr2tc orig_value = value;

  if (is_scalar_type(orig_value)) {
    if (type != orig_value->type) {
      if (base_type_eq(value->type, type, ns)) {
        value = typecast2tc(type, value);
      } else {
        // XXX -- uh, byte swapping? FIXME
        value = typecast2tc(type, value);
      }
    }
  } else if (is_array_type(orig_value) || is_string_type(orig_value)) {
    // We have zero offset. Just select things out.
    type2tc arr_subtype = (is_array_type(orig_value))
     ? to_array_type(orig_value->type).subtype
     : to_array_type(to_constant_string2t(orig_value).to_array()->type).subtype;
    unsigned int access_size_int = type->get_width() / 8;
    unsigned long subtype_size_int = type_byte_size(*arr_subtype).to_ulong();

    if (is_array_type(arr_subtype)) {
      construct_from_multidir_array(value, zero_uint, type, guard,
                                    scalar_step_list,
                                    config.ansi_c.word_size);
    } else if (is_structure_type(arr_subtype)) {
      value = index2tc(arr_subtype, orig_value, zero_uint);
      construct_from_const_struct_offset(value, zero_uint, type, guard);
    } else {
      assert(is_scalar_type(arr_subtype));

      // Now, if the subtype size is >= the read size, we can just either cast
      // or extract out. If not, we have to extract by conjoining elements.
      if (!is_big_endian && subtype_size_int >= access_size_int) {
        // Voila, one can just select and cast. This works because little endian
        // just allows for this to happen.
        index2tc idx(arr_subtype, orig_value, zero_uint);
        typecast2tc cast(type, idx);
        value = cast;
      } else {
        // Nope, one must byte extract this.
        const type2tc &bytetype = get_uint8_type();
        value = byte_extract2tc(bytetype, orig_value, zero_uint, is_big_endian);
        if (type != bytetype)
          value = typecast2tc(type, value);
      }
    }

    bounds_check(orig_value->type, zero_int, access_size_int, guard);

    //XXX uuhh, in desperate need of refactor.
    if (scalar_step_list->size() != 0 && !is_scalar_type(type))
      wrap_in_scalar_step_list(value, scalar_step_list, guard);
  } else {
    assert(is_structure_type(orig_value));
    assert(scalar_step_list != NULL);
    if (scalar_step_list->size() != 0) {
      // We have zero offset; If the base types here are compatible, then we can
      // just apply the set of scalar steps to this expr.

      // Fetch what's either the source of the index, or member, in the first
      // step.
      wrap_in_scalar_step_list(value, scalar_step_list, guard);
    } else {
      // No set of scalar steps: what this means is that we're accessing the
      // first element of this struct as it's natural type. Build the access
      // ourself.
      construct_from_const_struct_offset(value, zero_uint, type, guard);
    }
  }
}

void
dereferencet::construct_from_const_offset(expr2tc &value, const expr2tc &offset,
                                          const type2tc &type,
                                          const guardt &guard,
                                          std::list<expr2tc> *scalar_step_list __attribute__((unused)),
                                          bool checks)
{

  expr2tc base_object = value;

  const constant_int2t &theint = to_constant_int2t(offset);
  const type2tc &bytetype = get_uint8_type();

  if (is_array_type(base_object) || is_string_type(base_object)) {
    type2tc arr_subtype = (is_array_type(base_object))
    ? to_array_type(base_object->type).subtype
    : to_array_type(to_constant_string2t(base_object).to_array()->type).subtype;

    unsigned long subtype_size = type_byte_size(*arr_subtype).to_ulong();
    unsigned long deref_size = type->get_width() / 8;

    if (is_array_type(arr_subtype)) {
      construct_from_multidir_array(value, offset, type, guard,
                                    scalar_step_list,
                                    config.ansi_c.word_size);
    } else if (is_structure_type(arr_subtype)) {
      constant_int2tc subtype_sz_expr(index_type2(), BigInt(subtype_size));
      div2tc div(index_type2(), offset, subtype_sz_expr);
      modulus2tc mod(index_type2(), offset, subtype_sz_expr);

      expr2tc mod2 = mod->simplify();
      assert(is_constant_int2t(mod2) && "Modulus of constant offset should "
             "simplify to a constant");

      value = index2tc(arr_subtype, value, div);
      construct_from_const_struct_offset(value, mod2, type, guard);
    } else if (subtype_size == deref_size) {
      // We can just extract this, assuming it's aligned. If it's not aligned,
      // that's an error?
      expr2tc idx = offset;

      // Divide offset to an index if we're not an array of chars or something.
      if (subtype_size != 1) {
        constant_int2tc subtype_size_expr(offset->type, BigInt(subtype_size));
        div2tc index(offset->type, offset, subtype_size_expr);
        idx = index;
      }

      index2tc res(arr_subtype, base_object, idx);
      value = res;

      if (!base_type_eq(type, res->type, ns)) {
        // Wrong type but matching size; typecast.
        typecast2tc cast(type, value);
        value = cast;
      }
    } else if (subtype_size > deref_size) {
      // Firstly, is this an aligned access?
      if (subtype_size != 1) {
        unsigned long submask = subtype_size - 1;
        unsigned long res = theint.constant_value.to_ulong() & submask;
        if (res != 0) {
          dereference_callback.dereference_failure(
            "pointer dereference",
            "Unaligned access to non-byte array", guard);
          value = expr2tc();
          return;
        }
      }

      // Now, assuming it's aligned, just select an element then byte extract
      // from that.
      expr2tc idx_expr =
        gen_uint(theint.constant_value.to_ulong() / subtype_size);
      expr2tc mod_expr =
        gen_uint(theint.constant_value.to_ulong() % subtype_size);
      value = index2tc(arr_subtype, value, idx_expr);
      value = byte_extract2tc(get_uint_type(deref_size * 8), value, mod_expr,
                              is_big_endian);
      if (type != value->type)
        value = typecast2tc(type, value);
    } else {
      value = byte_extract2tc(bytetype, base_object, offset, is_big_endian);
      if (type->get_width() != 8)
        value = typecast2tc(type, value);
    }
  } else if (is_code_type(base_object)) {
    // Accessing anything but the start of a function is not permitted.
    notequal2tc neq(offset, zero_uint);
    if(!options.get_bool_option("no-pointer-check")) {
      guardt tmp_guard2(guard);
      tmp_guard2.add(false_expr);

      dereference_callback.dereference_failure(
        "Code separation",
        "Dereferencing code pointer with nonzero offset", tmp_guard2);
    }
  } else if (is_struct_type(base_object)) {
    // Just to be sure:
    assert(is_scalar_type(type));

    // Right. Hand off control to a specialsed function that goes through
    // structs recursively, determining what object we're operating on at
    // each point.
    construct_from_const_struct_offset(value, offset, type, guard);
  } else {
    assert(is_scalar_type(base_object));
    value = byte_extract2tc(bytetype, base_object, offset, is_big_endian);
    if (type->get_width() != 8)
      value = typecast2tc(type, value);
  }

  if (!checks)
    return;

  unsigned long access_sz =  type_byte_size(*type).to_ulong();
  if (is_array_type(base_object) || is_string_type(base_object)) {
    bounds_check(base_object->type, offset, access_sz, guard);
  } else {
    unsigned long sz = type_byte_size(*base_object->type).to_ulong();
    if (sz < theint.constant_value.to_ulong() + access_sz) {
      if(!options.get_bool_option("no-pointer-check")) {
        // This is statically known to be out of bounds.
        dereference_callback.dereference_failure(
          "pointer dereference",
          "Offset out of bounds", guard);
      }
    }
  }
}

void
dereferencet::construct_from_const_struct_offset(expr2tc &value,
                        const expr2tc &offset, const type2tc &type,
                        const guardt &guard)
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
      // middled. Which is an error.
      std::cerr << "Implement read-from-padding error in structs" << std::endl;
      abort();
    } else if (int_offset == m_offs) {
      // Does this over-read?
      if (access_size > m_size) {
        dereference_callback.dereference_failure(
          "pointer dereference",
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
        construct_from_const_offset(res, zero_uint, type, guard, NULL, false);
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
      construct_from_const_offset(memb, new_offs, type, guard, NULL, false);
      value = memb;
      return;
    } else if (int_offset < (m_offs + m_size)) {
      // This access starts in this field, but by process of elimination,
      // doesn't end in it. Which means reading padding data (or an alignment
      // error), which are both bad.
      dereference_callback.dereference_failure(
        "pointer dereference",
        "Misaligned access to struct field", guard);
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
    unsigned int word_mask = config.ansi_c.word_size - 1;
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
                                       alignment, &failed_container);
      extract_list.push_back(std::pair<expr2tc,expr2tc>(field_guard, field));
    } else if (access_sz > ((*it)->get_width() / 8)) {
      guardt newguard(guard);
      newguard.add(field_guard);
      dereference_callback.dereference_failure(
        "pointer dereference",
        "Oversized field offset", guard);
      // Push nothing back, allow fall-through of the if-then-else chain to
      // resolve to a failed deref symbol.
    } else if (alignment >= config.ansi_c.word_size) {
      // This is fully aligned, just pull it out and possibly cast,
      expr2tc field = member2tc(*it, value, struct_type.member_names[i]);
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
                                        const type2tc &type,
                                        const guardt &guard,
                                        unsigned long alignment,
                                        bool checks)
{
  assert(alignment != 0);
  unsigned long access_sz = type->get_width() / 8;
  expr2tc orig_value = value;

  // If the base thing is an array, and we have an appropriately aligned
  // reference, then just extract from it.
  if (is_array_type(value) || is_string_type(value)) {
    const array_type2t &arr_type = (is_array_type(value))
      ? to_array_type(value->type)
      : to_array_type(to_constant_string2t(value).to_array()->type);
    unsigned long subtype_sz = type_byte_size(*arr_type.subtype).to_ulong();

    if (is_array_type(arr_type.subtype)) {
      construct_from_multidir_array(value, offset, type, guard, NULL,alignment);
    } else if (is_structure_type(arr_type.subtype)) {
      mp_integer subtype_sz = type_byte_size(*arr_type.subtype);
      constant_int2tc subtype_sz_expr(index_type2(), subtype_sz);
      div2tc div(index_type2(), offset, subtype_sz_expr);
      modulus2tc mod(index_type2(), offset, subtype_sz_expr);

      value = index2tc(arr_type.subtype, value, div);
      construct_from_dyn_struct_offset(value, mod, type, guard, alignment);
    } else if (alignment >= subtype_sz && access_sz <= subtype_sz) {
      // Aligned access; just issue an index.
      expr2tc new_offset = offset;

      // If not an array of bytes or something, scale offset to index.
      if (subtype_sz != 1) {
        constant_int2tc subtype_sz_expr(offset->type, BigInt(subtype_sz));
        new_offset = div2tc(offset->type, offset, subtype_sz_expr);
      }

      index2tc idx(arr_type.subtype, value, new_offset);
      value = idx;
    } else {
      // Unstructured array access. First, check alignment.
      unsigned int subtype_sz = arr_type.subtype->get_width() / 8;
      if (subtype_sz != 1) {
        expr2tc align_expr = gen_uint(subtype_sz - 1);
        expr2tc masked_offs = bitand2tc(align_expr->type, align_expr, offset);
        expr2tc is_aligned = equality2tc(masked_offs, zero_uint);

        guardt tmp_guard = guard;
        tmp_guard.add(not2tc(is_aligned));
        dereference_callback.dereference_failure(
          "Pointer alignment",
          "Unaligned access to array", tmp_guard);
      }

      // From here on, assume aligned access.
      unsigned int target_bytes = type->get_width() / 8;
      expr2tc accuml;
      expr2tc accuml_offs = offset;
      type2tc subtype = arr_type.subtype;

      for (unsigned int i = 0; i < target_bytes; i++) {
        expr2tc elem = index2tc(subtype, value, accuml_offs);

        if (is_nil_expr(accuml)) {
          accuml = elem;
        } else {
          // XXX -- byte order.
          type2tc res_type = get_uint_type((i+1) * subtype_sz * 8);
          accuml = concat2tc(res_type, accuml, elem);
        }

        accuml_offs = add2tc(offset->type, accuml_offs, one_uint);
      }

      // That's going to come out as a bitvector;
      if (type != accuml->type) {
        // XXX -- we might be selecting a char out of an int array, or something
        // XXX -- byte order.
        //assert(type->get_width() == accuml->type->get_width());
        accuml = typecast2tc(type, accuml);
      }

      value = accuml;
    }

    if (checks)
      bounds_check(orig_value->type, offset, access_sz, guard);
    return;
  } else if (is_code_type(value)) {
    // No data is read out, we can only check for correctness here. And that
    // correctness demands that the offset is always zero.
    notequal2tc neq(offset, zero_uint);
    if(!options.get_bool_option("no-pointer-check")) {
      guardt tmp_guard2(guard);
      tmp_guard2.add(false_expr);

      dereference_callback.dereference_failure(
        "Code separation",
        "Dereferencing code pointer with nonzero offset", tmp_guard2);
    }

    return;
  }

  expr2tc new_offset = offset;
  if (memory_model(value, type, guard, new_offset))
  {
    // ok
  }
  else
  {
    if(!options.get_bool_option("no-pointer-check"))
    {
      //nec: ex29
      if (    (is_pointer_type(type) &&
               is_empty_type(to_pointer_type(type).subtype))
          ||
              (is_pointer_type(value) &&
               is_empty_type(to_pointer_type(value->type).subtype)))
        return;

      std::string msg="memory model not applicable (got `";
      msg+=from_type(ns, "", value->type);
      msg+="', expected `";
      msg+=from_type(ns, "", type);
      msg+="')";

      dereference_callback.dereference_failure(
        "pointer dereference",
        msg, guard);
    }

    value = expr2tc();
    return; // give up, no way that this is ok
  }
}

void dereferencet::valid_check(
  const expr2tc &object,
  const guardt &guard,
  const modet mode)
{
  if(options.get_bool_option("no-pointer-check"))
    return;

  const expr2tc &symbol = get_symbol(object);

  if (is_constant_string2t(symbol))
  {
    // always valid, but can't write

    if(mode==WRITE)
    {
      dereference_callback.dereference_failure(
        "pointer dereference",
        "write access to string constant",
        guard);
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
      dereference_callback.dereference_failure(
        "pointer dereference",
        "invalidated dynamic object",
        tmp_guard);
    } else {
      // Not dynamic; if we're in free mode, that's an error.
      if(mode==FREE)
      {
        dereference_callback.dereference_failure(
          "pointer dereference",
          "free() of non-dynamic memory",
          guard);
        return;
      }
    }
  }
}

void dereferencet::bounds_check(const type2tc &type, const expr2tc &offset,
                                unsigned int access_size, const guardt &guard)
{
  if(options.get_bool_option("no-bounds-check"))
    return;

  assert(is_array_type(type) || is_string_type(type));

  // Dance around getting the array type normalised.
  type2tc new_string_type;
  const array_type2t *arr_type_p = NULL;
  if (is_array_type(type)) {
    arr_type_p = &to_array_type(type);
  } else {
    const string_type2t &str_type = to_string_type(type);
    expr2tc str_size = gen_uint(str_type.width);
    new_string_type =
      type2tc(new array_type2t(get_uint8_type(), str_size, false));
    arr_type_p = &to_array_type(new_string_type);
  }

  const array_type2t &arr_type = *arr_type_p;

  // XXX --  arrays were assigned names, but we're skipping that for the moment
  // std::string name = array_name(ns, expr.source_value);

  // Firstly, bail if this is an infinite sized array. There are no bounds
  // checks to be performed.
  if (arr_type.size_is_infinite)
    return;

  // Secondly, try to calc the size of the array.
  unsigned long subtype_size_int = type_byte_size(*arr_type.subtype).to_ulong();
  constant_int2tc subtype_size(get_uint32_type(), BigInt(subtype_size_int));
  mul2tc arrsize(get_uint32_type(), arr_type.array_size, subtype_size);

  // Then, expressions as to whether the access is over or under the array
  // size.
  constant_int2tc access_size_e(get_uint32_type(), BigInt(access_size));
  add2tc upper_byte(get_uint32_type(), offset, access_size_e);
  expr2tc lower_byte = offset;

  greaterthan2tc gt(upper_byte, arrsize);
  lessthan2tc lt(lower_byte, zero_int);

  // Report these as assertions; they'll be simplified away if they're constant

  guardt tmp_guard1(guard);
  tmp_guard1.move(gt);
  dereference_callback.dereference_failure("array bounds", "array upper bound",
                                           tmp_guard1);

  guardt tmp_guard2(guard);
  tmp_guard2.move(lt);
  dereference_callback.dereference_failure("array bounds", "array lower bound",
                                           tmp_guard2);
}

bool dereferencet::memory_model(
  expr2tc &value,
  const type2tc &to_type,
  const guardt &guard,
  expr2tc &new_offset)
{
  // we will allow more or less arbitrary pointer type cast

  const type2tc &from_type = value->type;

  // first, check if it's really just a conversion

  if (is_bv_type(from_type) && is_bv_type(to_type) &&
      from_type->get_width() == to_type->get_width() &&
      is_constant_int2t(new_offset) &&
      to_constant_int2t(new_offset).constant_value.is_zero()) {
    value = typecast2tc(to_type, value);
    return true;
  }

  // otherwise, we will stich it together from bytes

  bool ret = memory_model_bytes(value, to_type, guard, new_offset);
  return ret;
}

bool dereferencet::memory_model_bytes(
  expr2tc &value,
  const type2tc &to_type,
  const guardt &guard,
  expr2tc &new_offset)
{
  const expr2tc orig_value = value;
  const type2tc from_type = value->type;

  // Accessing code is incorrect; The C spec says that the code and data address
  // spaces should be considered seperate (i.e., Harvard arch) and so accessing
  // code via a pointer is never valid. Even though you /can/ do it on X86.
  if (is_code_type(from_type) || is_code_type(to_type)) {
    guardt tmp_guard(guard);
    dereference_callback.dereference_failure("Code seperation",
        "Dereference accesses code / program text", tmp_guard);
    return true;
  }

  assert(config.ansi_c.endianess != configt::ansi_ct::NO_ENDIANESS);

  // We allow reading more or less anything as bit-vector.
  if (is_bv_type(to_type) || is_pointer_type(to_type) ||
        is_fixedbv_type(to_type))
  {
    // Take existing pointer offset, add to the pointer offset produced by
    // this dereference. It'll get simplified at some point in the future.
    new_offset = add2tc(new_offset->type, new_offset,
                        compute_pointer_offset(value));
    expr2tc tmp = new_offset->simplify();
    if (!is_nil_expr(tmp))
      new_offset = tmp;

    // XXX This isn't taking account of the additional offset being torn through
    expr2tc base_object = get_base_object(value);


    const type2tc &bytetype = get_uint8_type();
    value = byte_extract2tc(bytetype, base_object, new_offset, is_big_endian);

    // XXX jmorse - temporary, while byte extract is still covered in bees.
    value = typecast2tc(to_type, value);


    if (!is_constant_int2t(new_offset) ||
        !to_constant_int2t(new_offset).constant_value.is_zero())
    {
      if(!options.get_bool_option("no-pointer-check"))
      {
        // Get total size of the data object we're working on.
        expr2tc total_size;
        try {
          total_size = constant_int2tc(uint_type2(),
                                       base_object->type->get_width() / 8);
        } catch (array_type2t::dyn_sized_array_excp *e) {
          expr2tc eight = gen_uint(8);
          total_size = div2tc(uint_type2(), e->size, eight);
        }

        unsigned long width = to_type->get_width() / 8;
        expr2tc const_val = gen_uint(width);
        add2tc upper_bound(uint_type2(), new_offset, const_val);
        greaterthan2tc upper_bound_eq(upper_bound, total_size);

        guardt tmp_guard(guard);
        tmp_guard.move(upper_bound_eq);
        dereference_callback.dereference_failure(
            "byte model object boundries",
            "byte access upper bound", tmp_guard);

        lessthan2tc offs_lower_bound(new_offset, zero_int);

        guardt tmp_guard2(guard);
        tmp_guard2.move(offs_lower_bound);
        dereference_callback.dereference_failure(
          "byte model object boundries",
          "word offset lower bound", tmp_guard);
      }
    }

    return true;
  }

  return false;
}

unsigned int
dereferencet::fabricate_scalar_access(const type2tc &src_type,
                                      std::list<expr2tc> &scalar_step_list)
{
  // From the given type, insert a series of steps that reads out the first
  // element in some way that doesn't trigger a futher deference.
  assert(!is_scalar_type(src_type));
  // scalar_step_list may very well be empty, and that's fine.

  unsigned int steps = 0;
  type2tc cur_type = src_type;
  do {
    if (is_scalar_type(cur_type))
      break;

    if (is_array_type(cur_type)) {
      const array_type2t &arr_type = to_array_type(cur_type);

      // A safe access to an array is to select the first element -- C89 doesn't
      // allow for zero sized arrays (though GNUC does). XXX, this addition to
      // the scalar step list is esoteric. Also, contains a nil expr.
      index2tc idx(arr_type.subtype, expr2tc(), zero_uint);
      scalar_step_list.push_front(idx);
      cur_type = idx->type;
    } else if (is_struct_type(cur_type)) {
      const struct_type2t &struct_type = to_struct_type(cur_type);

      // Safe access -- select the first member of the struct. C89 doesn't allow
      // structs with no members.
      assert(struct_type.members.size() != 0 &&
             "C does not allow for zero sized structs");
      member2tc memb(struct_type.members[0], expr2tc(),
                     struct_type.member_names[0]);
      scalar_step_list.push_front(memb);
      cur_type = memb->type;
    } else {
      std::cerr << "Unrecognized type in fabricate_scalar_access" << std::endl;
      cur_type->dump();
      abort();
    }

    steps++;
  } while (true);

  return steps;
}

void
dereferencet::wrap_in_scalar_step_list(expr2tc &value,
                                       std::list<expr2tc> *scalar_step_list,
                                       const guardt &guard)
{
  // Check that either the base type that these steps are applied to matches
  // the type of the object we're wrapping in these steps. It's a type error
  // if there isn't a match.
  // Alternately, if the base expression is nil, then this was created by
  // fabricate_scalar_access, so be less strenuous.
  expr2tc base_of_steps = *scalar_step_list->front()->get_sub_expr(0);
  if (is_nil_expr(base_of_steps) ||
      dereference_type_compare(value, base_of_steps->type)) {
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
    dereference_callback.dereference_failure(
      "Memory model",
      "Object accessed with incompatible base type", guard);
    value = expr2tc();
  }
}

void
dereferencet::construct_from_multidir_array(expr2tc &value,
                              const expr2tc &offset,
                              const type2tc &type, const guardt &guard,
                              std::list<expr2tc> *scalar_step_list,
                              unsigned long alignment)
{
  assert(is_array_type(value) || is_string_type(value));
  const array_type2t &arr_type = (is_array_type(value))
    ? to_array_type(value->type)
    : to_array_type(to_constant_string2t(value).to_array()->type);

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

  if (is_constant_expr(idx)) {
    construct_from_const_offset(value, idx, type, guard, scalar_step_list, false);
  } else {
    construct_from_dyn_offset(value, idx, type, guard, alignment, false);
  }
}

void
dereferencet::construct_struct_ref_from_const_offset(expr2tc &value,
             const expr2tc &offs, const type2tc &type, const guardt &guard,
             std::list<expr2tc> *scalar_step_list)
{
  // Minimal effort: the moment that we can throw this object out due to an
  // incompatible type, we do.
  const constant_int2t &intref = to_constant_int2t(offs);

  if (is_array_type(value->type)) {
    const array_type2t &arr_type = to_array_type(value->type);

    if (!is_struct_type(arr_type.subtype) && !is_array_type(arr_type.subtype)) {
      // Can't handle accesses to anything else.
      dereference_callback.dereference_failure(
        "Memory model",
        "Object accessed with incompatible base type", guard);
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

    construct_struct_ref_from_const_offset(value, mod_expr, type, guard,
                                           scalar_step_list);
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
      } else {
        // Correctly aligned access to incompatible base type.
        dereference_callback.dereference_failure(
          "Memory model",
          "Object accessed with incompatible base type", guard);
        return;
      }
    } else {
      // If the offset isn't zero; enumerate through all the fields in this
      // struct, and work out whether or not it points into a sub-struct.
      const struct_type2t &struct_type = to_struct_type(value->type);
      unsigned int i = 0;
      forall_types(it, struct_type.members) {
        mp_integer offs = member_offset(struct_type,
                                        struct_type.member_names[i]);
        mp_integer size = type_byte_size(*(*it).get());

        if (intref.constant_value >= offs &&
              intref.constant_value <= (offs + size)) {
          // It's this field. Don't make a decision about whether it's correct
          // or not, recurse to make that happen.
          mp_integer new_offs = intref.constant_value - offs;
          expr2tc offs_expr = gen_uint(new_offs.to_ulong());
          value = member2tc(*it, value, struct_type.member_names[i]);
          construct_struct_ref_from_const_offset(value, offs_expr, type, guard,
                                                 scalar_step_list);
          return;
        }
        i++;
      }

      // Fell out of that loop. Either this offset is out of range, or lies in
      // padding.
      dereference_callback.dereference_failure(
        "Memory model",
        "Object accessed with illegal offset", guard);
    }
  } else {
    dereference_callback.dereference_failure(
      "Memory model",
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
    dereference_callback.dereference_failure(
      "Memory model",
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
  dereference_callback.dereference_failure(
    "Memory model",
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
