#include <sstream>

#include <ansi-c/c_types.h>

#include "smt_conv.h"

// So, the SMT-encoding-with-no-tuple-support. SMT itself doesn't support
// tuples, but we've been pretending that it does by using Z3 almost
// exclusively (which does support tuples). So, we have to find some way of
// supporting:
//
//  1) Tuples
//  2) Arrays of tuples
//  3) Arrays of tuples containing arrays
//
// 1: In all circumstances we're either creating, projecting, or updating,
//    a set of variables that are logically grouped together as a tuple. We
//    can generally just handle that by linking up a tuple to the actual
//    variable that we want. To do this, we create a set of symbols /underneath/
//    the symbol we're dealing with, corresponding to the field name. So a tuple
//    with fields a, b, and c, with the symbol name "faces" would create:
//
//      c::main::1::faces.a
//      c::main::1::faces.b
//      c::main::1::faces.c
//
//    As variables with the appropriate type. Project / update redirects
//    expressions to deal with those symbols. Equality is similar.
//
//    It gets more complicated with ite's though, as you can't
//    nondeterministically switch between symbol prefixes like that. So instead
//    we create a fresh new symbol, and create an ite for each member of the
//    tuple, binding it into the new fresh symbol if it's enabling condition
//    is true.
//
//    The basic design feature here is that in all cases where we have something
//    of tuple type, the AST is actually a deterministic symbol that we hang
//    additional bits of name off as appropriate.
//
//  2: For arrays of tuples, we do almost exactly as above, but all the new
//     variables that are used are in fact arrays of values. Then when we
//     perform an array operation, we either select the relevant values out into
//     a fresh tuple, or decompose a tuple into a series of updates to make.
//
//  3: Tuples of arrays of tuples are currently unimplemented. But it's likely
//     that we'll end up following 2: above, but extending the domain of the
//     array to contain the outer and inner array index. Ugly but works
//     (inspiration came from the internet / stackoverflow, reading other
//     peoples work where they've done this).
//
// All these things could probably be made more efficient by rewriting the
// expressions that are being converted, so that we can for example discard
// any un-necessary equalities or assertions. However, in the meantime, this
// slower approach works.

__attribute__((always_inline)) static inline const tuple_smt_ast *
to_tuple_ast(const smt_ast *a)
{
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL && "Tuple AST mismatch");
  return ta;
}

__attribute__((always_inline)) static inline const tuple_smt_sort *
to_tuple_sort(const smt_sort *a)
{
  const tuple_smt_sort *ta = dynamic_cast<const tuple_smt_sort *>(a);
  assert(ta != NULL && "Tuple AST mismatch");
  return ta;
}

smt_ast *
smt_convt::tuple_create(const expr2tc &structdef)
{
  // From a vector of expressions, create a tuple representation by creating
  // a fresh name and assigning members into it.
  std::string name = mk_fresh_name("tuple_create::");

  const smt_ast *args[structdef->get_num_sub_exprs()];
  for (unsigned int i = 0; i < structdef->get_num_sub_exprs(); i++)
    args[i] = convert_ast(*structdef->get_sub_expr(i));

  tuple_create_rec(name, structdef->type, args);

  return new tuple_smt_ast(convert_sort(structdef->type), name);
}

smt_ast *
smt_convt::tuple_fresh(const smt_sort *s)
{
  std::string name = mk_fresh_name("tuple_fresh::");

  smt_ast *a = mk_smt_symbol(name, s);
  a = a;
  return new tuple_smt_ast(s, name);
}

const struct_union_data &
smt_convt::get_type_def(const type2tc &type)
{

  return (is_pointer_type(type))
        ? *pointer_type_data
        : dynamic_cast<const struct_union_data &>(*type.get());
}

void
smt_convt::tuple_create_rec(const std::string &name, const type2tc &structtype,
                            const smt_ast **inputargs)
{
  // Iterate over the members of a struct; if a member is a struct itself,
  // recurse, otherwise compute the name of the field and assign in the value
  // of that field.
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &data = get_type_def(structtype);

  unsigned int i = 0;
  forall_types(it, data.members) {
    if (is_tuple_ast_type(*it) || is_tuple_array_ast_type(*it)) {
      // This is a complicated field, but when the ast for this tuple / array
      // was converted, we already created an appropriate tuple_smt_ast, so we
      // just need to equality it into the tuple with the passed in name.
      std::string subname = name + data.member_names[i].as_string() + ".";
      const tuple_smt_ast *target =
        new tuple_smt_ast(convert_sort(*it), subname);
      const smt_ast *src = inputargs[i];

      if (is_tuple_ast_type(*it))
        assert_lit(mk_lit(tuple_equality(target, src)));
      else
        assert_lit(mk_lit(tuple_array_equality(target, src)));
    } else {
      // This is a normal field -- take the value from the inputargs array,
      // compute the members name, and then make an equality.
      std::string symname = name + data.member_names[i].as_string();
      const smt_sort *sort = convert_sort(*it);
      const smt_ast *args[2];
      args[0] = mk_smt_symbol(symname, sort);
      args[1] = inputargs[i];
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
    }

    i++;
  }
}

smt_ast *
smt_convt::mk_tuple_symbol(const expr2tc &expr)
{
  // Assuming this is a symbol, convert it to being an ast with tuple type.
  // That's done by creating the prefix for the names of all the contained
  // variables, and storing it.
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + ".";
  const smt_sort *sort = convert_sort(sym.type);
  return new tuple_smt_ast(sort, name);
}

smt_ast *
smt_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  // Exactly the same as creating a tuple symbol, but for arrays.
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + "[]";
  const smt_sort *sort = convert_sort(sym.type);
  return new tuple_smt_ast(sort, name);
}

smt_ast *
smt_convt::tuple_project(const smt_ast *a, const smt_sort *s, unsigned int i)
{
  // Create an AST representing the i'th field of the tuple a. This means we
  // have to open up the (tuple symbol) a, tack on the field name to the end
  // of that name, and then return that. It now names the variable that contains
  // the value of that field. If it's actually another tuple, we instead return
  // a new tuple_smt_ast containing its name.
  const tuple_smt_ast *ta = to_tuple_ast(a);
  const tuple_smt_sort *ts = to_tuple_sort(a->sort);
  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  assert(i < data.members.size() && "Out-of-bounds tuple element accessed");
  const std::string &fieldname = data.member_names[i].as_string();
  std::string sym_name = ta->name + fieldname;

  // Cope with recursive structs.
  const type2tc &restype = data.members[i];
  if (is_tuple_ast_type(restype) || is_tuple_array_ast_type(restype)) {
    // This is a struct within a struct, so just generate the name prefix of
    // the internal struct being projected.
    sym_name = sym_name + ".";
    return new tuple_smt_ast(s, sym_name);
  } else {
    // This is a normal variable, so create a normal symbol of its name.
    return mk_smt_symbol(sym_name, s);
  }
}

const smt_ast *
smt_convt::tuple_update(const smt_ast *a, unsigned int i, const smt_ast *v)
{
  // Take the tuple_smt_ast a and update the ith field with the value v. As
  // ever, we do this by creating a new tuple. The non-ith values are just
  // assigned into the new tuple, and the ith member is replaced with v.
  const smt_ast *args[2];
  bvt eqs;
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  // Create a fresh tuple to store the result in
  std::string name = mk_fresh_name("tuple_update::");
  const tuple_smt_ast *result = new tuple_smt_ast(a->sort, name);
  const tuple_smt_ast *ta = to_tuple_ast(a);
  const tuple_smt_sort *ts = to_tuple_sort(ta->sort);
  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  // Iterate over all members, deciding what to do with them.
  unsigned int j = 0;
  forall_types(it, data.members) {
    if (j == i) {
      // This is the updated field -- generate the name of its variable with
      // tuple project and assign it in.
      const smt_sort *tmp = convert_sort(*it);
      const smt_ast *thefield = tuple_project(result, tmp, j);
      if (is_tuple_ast_type(*it)) {
        // If it's of tuple type though, we need to generate a tuple equality.
        eqs.push_back(mk_lit(tuple_equality(thefield, v)));
      } else {
        args[0] = thefield;
        args[1] = v;
        eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    } else {
      // This is not an updated field; extract the member out of the input
      // tuple (a) and assign it into the fresh tuple.
      if (is_tuple_ast_type(*it)) {
        // A tuple equality is required for tuples.
        std::stringstream ss2;
        ss2 << name << data.member_names[j] << ".";
        std::string field_name = name;
        const smt_sort *tmp = convert_sort(*it);
        const smt_ast *field1 = tuple_project(ta, tmp, j);
        const smt_ast *field2 = tuple_project(result, tmp, j);
        eqs.push_back(mk_lit(tuple_equality(field1, field2)));
      } else {
        const smt_sort *tmp = convert_sort(*it);
        args[0] = tuple_project(ta, tmp, j);
        args[1] = tuple_project(result, tmp, j);
        eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    }

    j++;
  }

  // Assert all the equalities we just generated.
  assert_lit(land(eqs));
  return result;
}

const smt_ast *
smt_convt::tuple_equality(const smt_ast *a, const smt_ast *b)
{
  // We have two tuple_smt_asts and need to create a boolean ast representing
  // their equality: iterate over all their members, compute an equality for
  // each of them, and then combine that into a final ast.
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const tuple_smt_ast *ta = to_tuple_ast(a);
  const tuple_smt_sort *ts = to_tuple_sort(ta->sort);
  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  std::vector<literalt> lits;
  lits.reserve(data.members.size());

  // Iterate through each field and encode an equality.
  unsigned int i = 0;
  forall_types(it, data.members) {
    if (is_tuple_ast_type(*it) || is_tuple_array_ast_type(*it)) {
      // Recurse.
      const smt_sort *sort = convert_sort(*it);
      const smt_ast *side1 = tuple_project(a, sort, i);
      const smt_ast *side2 = tuple_project(b, sort, i);

      literalt l;
      if (is_tuple_ast_type(*it))
        l = mk_lit(tuple_equality(side1, side2));
      else
        l = mk_lit(tuple_array_equality(side1, side2));

      lits.push_back(l);
    } else {
      // This is a normal piece of data, project it to get a normal smt symbol
      // and encode an equality between the two values.
      const smt_ast *args[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = tuple_project(a, sort, i);
      args[1] = tuple_project(b, sort, i);
      const smt_ast *eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(eq);
      lits.push_back(l);
    }

    i++;
  }

  // Create an ast representing the fact that all the members are equal.
  literalt l = land(lits);
  return lit_to_ast(l);
}

const smt_ast *
smt_convt::tuple_ite(const smt_ast *cond, const smt_ast *true_val,
                     const smt_ast *false_val, const smt_sort *sort)
{
  // Prepare to create an ite between our arguments; the heavy lifting is done
  // by tuple_ite_rec, here we generate a new name for these things to be stored
  // into, then pass everything down to tuple_ite_rec.
  const tuple_smt_ast *trueast = to_tuple_ast(true_val);
  const tuple_smt_ast *falseast = to_tuple_ast(false_val);

  // Create a fresh tuple to store the result in
  std::string name = mk_fresh_name("tuple_ite::");
  const tuple_smt_ast *result = new tuple_smt_ast(sort, name);

  tuple_ite_rec(result, cond, trueast, falseast);
  return result;
}

void
smt_convt::tuple_ite_rec(const tuple_smt_ast *result, const smt_ast *cond,
                         const tuple_smt_ast *true_val,
                         const tuple_smt_ast *false_val)
{
  // So - we need to generate an ite between true_val and false_val, that gets
  // switched on based on cond, and store the output into result. Do this by
  // projecting each member out of our arguments and computing another ite
  // over each member. Note that we always make assertions here, because the
  // ite is always true, we don't return anything.
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const tuple_smt_sort *ts = to_tuple_sort(true_val->sort);
  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  // Iterate through each field and encode an ite.
  unsigned int i = 0;
  forall_types(it, data.members) {
    if (is_tuple_ast_type(*it) || is_tuple_array_ast_type(*it)) {
      // Recurse.
      const tuple_smt_ast *args[3];
      const smt_sort *sort = convert_sort(*it);
      args[0] = to_tuple_ast(tuple_project(result, sort, i));
      args[1] = to_tuple_ast(tuple_project(true_val, sort, i));
      args[2] = to_tuple_ast(tuple_project(false_val, sort, i));

      if (is_tuple_ast_type(*it)) {
        tuple_ite_rec(args[0], cond, args[1], args[2]);
      } else {
        const tuple_smt_sort *tsort = to_tuple_sort(args[1]->sort);
        const array_type2t &array_type = to_array_type(*it);
        const smt_sort *dom_sort = make_array_domain_sort(array_type);
        tuple_array_ite_rec(args[1], args[2], cond, tsort->thetype, dom_sort,
                            args[0]);
      }
    } else {
      // Normal field: create symbols for the member in each of the arguments,
      // then create an ite between them, and assert it.
      const smt_ast *args[3], *eqargs[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = cond;
      args[1] = tuple_project(true_val, sort, i);
      args[2] = tuple_project(false_val, sort, i);
      eqargs[0] = mk_func_app(sort, SMT_FUNC_ITE, args, 3);
      eqargs[1] = tuple_project(result, sort, i);
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, eqargs, 2)));
    }

    i++;
  }
}

const smt_ast *
smt_convt::tuple_array_create(const type2tc &array_type,
                              const smt_ast **inputargs,
                              bool const_array,
                              const smt_sort *domain __attribute__((unused)))
{
  // Create a tuple array from a constant representation. This means that
  // either we have an array_of or a constant_array. Handle this by creating
  // a fresh tuple array symbol, then repeatedly updating it with tuples at each
  // index. Ignore infinite arrays, they're "not for you".
  // XXX - probably more efficient to update each member array, but not now.
  const smt_sort *sort = convert_sort(array_type);
  std::string name = mk_fresh_name("tuple_array_create::");
  const smt_ast *newsym = new tuple_smt_ast(sort, name);

  // Check size
  const array_type2t &arr_type = to_array_type(array_type);
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return newsym;
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const smt_sort *fieldsort = convert_sort(arr_type.subtype);
  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.constant_value.to_ulong();

  if (const_array) {
    // Repeatedly store the same value into this at all the demanded
    // indexes.
    const smt_ast *init = inputargs[0];
    for (unsigned int i = 0; i < sz; i++) {
      constant_int2tc idx(index_type2(), BigInt(i));
      newsym = tuple_array_update(newsym, idx, init, fieldsort);
    }

    return newsym;
  } else {
    // Repeatedly store operands into this.
    for (unsigned int i = 0; i < sz; i++) {
      constant_int2tc idx(index_type2(), BigInt(i));
      newsym = tuple_array_update(newsym, idx, inputargs[i], fieldsort);
    }

    return newsym;
  }
}

const smt_ast *
smt_convt::tuple_array_select(const smt_ast *a, const smt_sort *s,
                              const expr2tc &field)
{
  // Select everything at the given element into a fresh tuple. Don't attempt
  // to support selecting array fields. In the future we can arrange something
  // whereby tuple operations are aware of this array situation and don't
  // have to take this inefficient approach.
  const tuple_smt_ast *ta = to_tuple_ast(a);
  const tuple_smt_sort *ts = to_tuple_sort(a->sort);

  std::string name = mk_fresh_name("tuple_array_select::");
  const tuple_smt_ast *result = new tuple_smt_ast(s, name);

  const array_type2t &array_type = to_array_type(ts->thetype);
  tuple_array_select_rec(ta, array_type.subtype, result, field,
                         array_type.array_size);
  return result;
}

void
smt_convt::tuple_array_select_rec(const tuple_smt_ast *ta,
                                  const type2tc &subtype,
                                  const tuple_smt_ast *result,
                                  const expr2tc &field,
                                  const expr2tc &arr_width)
{
  // Implementation of selecting out of a tuple array: ta contains the source
  // tuple array, and result is a new tuple that we're assigning things into.
  // For each member of the tuple, create the member name from the tuple array 
  // to get an array variable, and then index that variable to actually perform
  // the select operation.
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = get_type_def(subtype);

  unsigned int i = 0;
  // For each member...
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
      // If it's a tuple itself, we have to recurse.
      const smt_sort *sort = convert_sort(*it);
      const tuple_smt_ast *result_field =
        to_tuple_ast(tuple_project(result, sort, i));
      std::string substruct_name =
        ta->name + struct_type.member_names[i].as_string() + ".";
      const tuple_smt_ast *array_name = new tuple_smt_ast(sort, substruct_name);
      tuple_array_select_rec(array_name, *it, result_field, field, arr_width);
    } else {
      // Otherwise assume it's a normal variable: create its name (which is of
      // array type), and then extract the value from that array at the
      // specified index.
      std::string name = ta->name + struct_type.member_names[i].as_string();
      type2tc this_arr_type(new array_type2t(*it, arr_width, false));
      const smt_ast *args[2];
      const smt_sort *field_sort = convert_sort(*it);
      symbol2tc array_sym(this_arr_type, name);
      expr2tc tmpidx = fix_array_idx(field, this_arr_type);
      args[0] = mk_select(array_sym, tmpidx, field_sort);
      args[1] = tuple_project(result, field_sort, i);
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
    }

    i++;
  }
}

const smt_ast *
smt_convt::tuple_array_update(const smt_ast *a, const expr2tc &index,
                              const smt_ast *val,
                              const smt_sort *fieldsort __attribute__((unused)))
{
  // Like tuple array select, but backwards: create a fresh new tuple array,
  // and assign into each member of it the array of values from the source,
  // but with the specified index updated. Here, we create the fresh value,
  // then recurse on it.
  const tuple_smt_ast *ta = to_tuple_ast(a);
  const tuple_smt_ast *tv = to_tuple_ast(val);
  const tuple_smt_sort *ts = to_tuple_sort(ta->sort);

  std::string name = mk_fresh_name("tuple_array_update[]::");
  const tuple_smt_ast *result = new tuple_smt_ast(a->sort, name);

  const array_type2t &arrtype = to_array_type(ts->thetype);
  const array_type2t &array_type = to_array_type(ts->thetype);
  tuple_array_update_rec(ta, tv, index, result, arrtype.array_size,
                         array_type.subtype);
  return result;
}

void
smt_convt::tuple_array_update_rec(const tuple_smt_ast *ta,
                                  const tuple_smt_ast *tv,
                                  const expr2tc &idx,
                                  const tuple_smt_ast *result,
                                  const expr2tc &arr_width,
                                  const type2tc &subtype)
{
  // Implementation of tuple array update: for each member take its array
  // variable from the source (in ta), then update it with the relevant member
  // of tv at index idx. Then assign that value into result.
  const struct_union_data &struct_type = get_type_def(subtype);

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
      // This is a struct; we need to do recurse again.
      const smt_sort *tmp = convert_sort(*it);
      std::string resname = result->name +
                            struct_type.member_names[i].as_string() +
                            ".";
      std::string srcname = ta->name + struct_type.member_names[i].as_string() +
                            ".";
      std::string valname = tv->name + struct_type.member_names[i].as_string() +
                            ".";
      const tuple_smt_ast *target = new tuple_smt_ast(tmp, resname);
      const tuple_smt_ast *src = new tuple_smt_ast(tmp, srcname);
      const tuple_smt_ast *val = new tuple_smt_ast(tmp, valname);

      tuple_array_update_rec(src, val, idx, target, arr_width, *it);
    } else {
      // Normal value; name, update, assign.
      std::string arrname = ta->name + struct_type.member_names[i].as_string();
      std::string valname = tv->name + struct_type.member_names[i].as_string();
      std::string resname = result->name +
                            struct_type.member_names[i].as_string();
      type2tc this_arr_type(new array_type2t(*it, arr_width, false));
      // Take the source array variable and update it into an ast.
      symbol2tc arrsym(this_arr_type, arrname);
      expr2tc tmp_idx = fix_array_idx(idx, this_arr_type);
      symbol2tc valsym(*it, valname);
      with2tc store(this_arr_type, arrsym, tmp_idx, valsym);

      // Now assign that ast into the result tuple array.
      symbol2tc ressym(this_arr_type, irep_idt(resname));
      const smt_ast *eq = convert_array_equality(ressym, store);
      assert_lit(mk_lit(eq));
    }

    i++;
  }
}

const smt_ast *
smt_convt::tuple_array_equality(const smt_ast *a, const smt_ast *b)
{
  // Almost exactly the same as tuple equality, but all the types are arrays
  // instead of their normal types.
  const tuple_smt_ast *ta = to_tuple_ast(a);
  const tuple_smt_ast *tb = to_tuple_ast(b);
  const tuple_smt_sort *ts = to_tuple_sort(a->sort);

  const array_type2t &array_type = to_array_type(ts->thetype);
  return tuple_array_equality_rec(ta, tb, array_type.array_size,
                                  array_type.subtype);
}

const smt_ast *
smt_convt::tuple_array_equality_rec(const tuple_smt_ast *a,
                                    const tuple_smt_ast *b,
                                    const expr2tc &arr_width,
                                    const type2tc &subtype)
{
  // Same as tuple equality rec, but with arrays instead of their normal types.
  bvt eqs;
  const struct_union_data &struct_type = get_type_def(subtype);

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
      // Recurse, as ever.
      const smt_sort *tmp = convert_sort(*it);
      std::string name1 = a->name + struct_type.member_names[i].as_string()+".";
      std::string name2 = b->name + struct_type.member_names[i].as_string()+".";
      const tuple_smt_ast *new1 = new tuple_smt_ast(tmp, name1);
      const tuple_smt_ast *new2 = new tuple_smt_ast(tmp, name2);
      eqs.push_back(mk_lit(tuple_array_equality_rec(new1, new2, arr_width,
                                                    *it)));
    } else {
      // Normal equality between members (which are in fact arrays).
      std::string name1 = a->name + struct_type.member_names[i].as_string();
      std::string name2 = b->name + struct_type.member_names[i].as_string();
      type2tc arrtype(new array_type2t(*it, arr_width, false));
      symbol2tc arr1(arrtype, irep_idt(name1));
      symbol2tc arr2(arrtype, irep_idt(name2));
      const smt_ast *eq = convert_array_equality(arr1, arr2);
      eqs.push_back(mk_lit(eq));
    }

    i++;
  }

  return lit_to_ast(land(eqs));
}

const smt_ast *
smt_convt::tuple_array_ite(const smt_ast *cond, const smt_ast *trueval,
                           const smt_ast *false_val,
                           const smt_sort *sort __attribute__((unused)))
{
  // Same deal as tuple_ite, but with array types. In this function we create
  // the fresh tuple array in which to store all the results into.
  const tuple_smt_ast *tv = to_tuple_ast(trueval);
  const tuple_smt_ast *fv = to_tuple_ast(false_val);
  const tuple_smt_sort *ts = to_tuple_sort(tv->sort);

  std::string name = mk_fresh_name("tuple_array_ite[]::");
  const tuple_smt_ast *result = new tuple_smt_ast(tv->sort, name);

  const array_type2t &array_type = to_array_type(ts->thetype);
  const smt_sort *dom_sort = make_array_domain_sort(array_type);
  tuple_array_ite_rec(tv, fv, cond, ts->thetype, dom_sort, result);
  return result;
}

void
smt_convt::tuple_array_ite_rec(const tuple_smt_ast *tv, const tuple_smt_ast *fv,
                               const smt_ast *cond, const type2tc &type,
                               const smt_sort *dom_sort,
                               const tuple_smt_ast *res)
{
  // Almost the same as tuple_ite, but with array types. Iterate over each
  // member of the type we're dealing with, projecting the members out then
  // computing an ite over each of them, storing into res.
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const array_type2t &array_type = to_array_type(type);
  const struct_union_data &struct_type = get_type_def(array_type.subtype);

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it) || is_tuple_array_ast_type(*it)) {
      const smt_sort *tmp = convert_sort(*it);
      std::string resname = res->name +
                            struct_type.member_names[i].as_string() +
                            ".";
      std::string truename= tv->name + struct_type.member_names[i].as_string() +
                            ".";
      std::string falsename=fv->name + struct_type.member_names[i].as_string() +
                            ".";
      const tuple_smt_ast *resval = new tuple_smt_ast(tmp, resname);
      const tuple_smt_ast *trueval = new tuple_smt_ast(tmp, truename);
      const tuple_smt_ast *falseval = new tuple_smt_ast(tmp, falsename);

      if (is_tuple_ast_type(*it)) {
        // Create an array type for this -- because it's contained in an array,
        // at the underlying level it's an array.
        type2tc tmp_arr_type(new array_type2t(*it, array_type.array_size,
                                              array_type.size_is_infinite));
        tuple_array_ite_rec(trueval, falseval, cond, tmp_arr_type,
                            dom_sort, resval);
      } else {
        tuple_array_ite_rec(trueval, falseval, cond, *it, dom_sort, resval);
      }
    } else {
      std::string tname = tv->name + struct_type.member_names[i].as_string();
      std::string fname = fv->name + struct_type.member_names[i].as_string();
      std::string rname = res->name + struct_type.member_names[i].as_string();
      const smt_ast *args[3];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);
      args[0] = cond;
      args[1] = mk_smt_symbol(tname, arrsort);
      args[2] = mk_smt_symbol(fname, arrsort);
      args[0] = mk_func_app(arrsort, SMT_FUNC_ITE, args, 3);
      args[1] = mk_smt_symbol(rname, arrsort);
      assert_lit(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
    }

    i++;
  }
}

expr2tc
smt_convt::tuple_get(const expr2tc &expr)
{
  assert(is_symbol2t(expr) && "Non-symbol in smtlib expr get()");
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name();

  const type2tc &thetype = (is_structure_type(expr->type))
    ? expr->type : pointer_struct;
  const struct_union_data &strct =
    static_cast<const struct_union_data &>(*thetype.get());

  // XXX - what's the correct type to return here.
  constant_struct2tc outstruct(expr->type, std::vector<expr2tc>());

  // Run through all fields and despatch to 'get' again.
  unsigned int i = 0;
  forall_types(it, strct.members) {
    std::stringstream ss;
    ss << name << "." << strct.member_names[i];
    symbol2tc sym(*it, ss.str());
    outstruct.get()->datatype_members.push_back(get(sym));
    i++;
  }

  return outstruct;
}

const smt_ast *
smt_convt::array_create(const expr2tc &expr)
{
  if (is_constant_array_of2t(expr))
    return convert_array_of_prep(expr);

  // Handle constant array expressions: these don't have tuple type and so
  // don't need funky handling, but we need to create a fresh new symbol and
  // repeatedly store the desired data into it, to create an SMT array
  // representing the expression we're converting.
  std::string name = mk_fresh_name("array_create::");
  expr2tc newsym = symbol2tc(expr->type, name);

  // Check size
  const array_type2t &arr_type =
    static_cast<const array_type2t &>(*expr->type.get());
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return convert_ast(newsym);
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.constant_value.to_ulong();

  assert(is_constant_array2t(expr));
  const constant_array2t &array = to_constant_array2t(expr);

  // Repeatedly store things into this.
  for (unsigned int i = 0; i < sz; i++) {
    constant_int2tc field(
                      type2tc(new unsignedbv_type2t(config.ansi_c.int_width)),
                      BigInt(i));
    expr2tc init = array.datatype_members[i];

    if (is_bool_type(array.datatype_members[i]->type) && !int_encoding &&
        no_bools_in_arrays)
      init = typecast2tc(type2tc(new unsignedbv_type2t(1)),
                         array.datatype_members[i]);

    newsym = with2tc(newsym->type, newsym, field, init);
  }

  return convert_ast(newsym);
}

const smt_ast *
smt_convt::convert_array_of_prep(const expr2tc &expr)
{
  const constant_array_of2t &arrof = to_constant_array_of2t(expr);
  const array_type2t &arrtype = to_array_type(arrof.type);
  expr2tc base_init;
  unsigned long array_size = 0;

  // So: we have an array_of, that we have to convert into a bunch of stores.
  // However, it might be a nested array. If that's the case, then we're
  // guarenteed to have another array_of in the initializer (which in turn might
  // be nested). In that case, flatten to a single array of whatever's at the
  // bottom of the array_of.
  if (is_array_type(arrtype.subtype)) {
    type2tc flat_type = flatten_array_type(expr->type);
    const array_type2t &arrtype2 = to_array_type(flat_type);
    array_size = calculate_array_domain_width(arrtype2);

    expr2tc rec_expr = expr;
    while (is_constant_array_of2t(rec_expr))
      rec_expr = to_constant_array_of2t(rec_expr).initializer;

    base_init = rec_expr;
  } else {
    base_init = arrof.initializer;
    array_size = calculate_array_domain_width(arrtype);
  }

  if (is_structure_type(base_init->type))
    return tuple_array_of(base_init, array_size);
  else if (is_pointer_type(base_init->type))
    return pointer_array_of(base_init, array_size);
  else
    return convert_array_of(base_init, array_size);
}

const smt_ast *
smt_convt::convert_array_of(const expr2tc &init_val, unsigned long array_size)
{
  // We now an initializer, and a size of array to build. So:

  std::vector<expr2tc> array_of_inits;
  for (unsigned long i = 0; i < (1ULL << array_size); i++)
    array_of_inits.push_back(init_val);

  constant_int2tc real_arr_size(index_type2(), BigInt(1ULL << array_size));
  type2tc newtype(new array_type2t(init_val->type, real_arr_size, false));

  expr2tc res(new constant_array2t(newtype, array_of_inits));
  return convert_ast(res);
}

const smt_ast *
smt_convt::tuple_array_of(const expr2tc &init_val, unsigned long array_size)
{
  assert(!tuple_support);

  // An array of tuples without tuple support: decompose into array_of's each
  // subtype.
  const struct_union_data &subtype =  get_type_def(init_val->type);
  const constant_datatype_data &data =
    static_cast<const constant_datatype_data &>(*init_val.get());

  constant_int2tc arrsize(index_type2(), BigInt(array_size));
  type2tc arrtype(new array_type2t(init_val->type, arrsize, false));
  const smt_sort *sort = convert_sort(arrtype);
  std::string name = mk_fresh_name("tuple_array_of::");
  const smt_ast *newsym = new tuple_smt_ast(sort, name);
  const smt_sort *bool_sort = mk_sort(SMT_SORT_BOOL);

  assert(subtype.members.size() == data.datatype_members.size());
  for (unsigned long i = 0; i < subtype.members.size(); i++) {
    const expr2tc &val = data.datatype_members[i];
    const smt_ast *sub_arr_of = convert_array_of(val, array_size);
    const smt_ast *args[2];

    const smt_sort *this_sort = convert_sort(subtype.members[i]);
    args[0] = tuple_project(newsym, this_sort, i);
    args[1] = sub_arr_of;
    assert_lit(mk_lit(mk_func_app(bool_sort, SMT_FUNC_EQ, args, 2)));
  }

  return newsym;
}

const smt_ast *
smt_convt::pointer_array_of(const expr2tc &init_val, unsigned long array_width)
{
  // Actually a tuple, but the operand is going to be a symbol, null.
  assert(is_symbol2t(init_val) && "Pointer type'd array_of can only be an "
         "array of null");
  const symbol2t &sym = to_symbol2t(init_val);
  assert(sym.thename == "NULL" && "Pointer type'd array_of can only be an "
         "array of null");

  // Well known value; zero and zero.
  constant_int2tc zero_val(machine_ptr, BigInt(0));
  std::vector<expr2tc> operands;
  operands.reserve(2);
  operands.push_back(zero_val);
  operands.push_back(zero_val);

  constant_struct2tc strct(pointer_struct, operands);
  return tuple_array_of(strct, array_width);
}

const smt_ast *
smt_convt::tuple_array_create_despatch(const expr2tc &expr,
                                       const smt_sort *domain)
{
  // Take a constant_array2t or an array_of, and format the data from them into
  // a form palatable to tuple_array_create.

  if (is_constant_array_of2t(expr)) {
    const constant_array_of2t &arr = to_constant_array_of2t(expr);
    const smt_ast *arg = convert_ast(arr.initializer);

    return tuple_array_create(arr.type, &arg, true, domain);
  } else {
    assert(is_constant_array2t(expr));
    const constant_array2t &arr = to_constant_array2t(expr);
    const smt_ast *args[arr.datatype_members.size()];
    unsigned int i = 0;
    forall_exprs(it, arr.datatype_members) {
      args[i] = convert_ast(*it);
      i++;
    }

    return tuple_array_create(arr.type, args, false, domain);
  }
}

const smt_ast *
smt_convt::convert_array_equality(const expr2tc &a, const expr2tc &b)
{
  const smt_ast *args[2];
  const smt_sort * s = mk_sort(SMT_SORT_BOOL);
  args[0] = convert_ast(a);
  args[1] = convert_ast(b);

  return mk_func_app(s, SMT_FUNC_EQ, args, 2);
}
