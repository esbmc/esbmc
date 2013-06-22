#include <sstream>

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

smt_ast *
smt_convt::tuple_create(const expr2tc &structdef)
{
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

void
smt_convt::tuple_create_rec(const std::string &name, const type2tc &structtype,
                            const smt_ast **inputargs)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &data = (is_pointer_type(structtype))
    ? *pointer_type_data
    : dynamic_cast<const struct_union_data &>(*structtype.get());

  unsigned int i = 0, j;
  forall_types(it, data.members) {
    if (is_tuple_ast_type(*it)) {
      // Do something recursive
      std::string subname = name + data.member_names[i].as_string() + ".";
      // Generate an array of fields to pump in. First, fetch the type. It has
      // to be something struct based.
      const struct_union_data &nextdata = (is_pointer_type(*it))
        ? *pointer_type_data
        : dynamic_cast<const struct_union_data &>(*(*it).get());
      const smt_ast *nextargs[nextdata.members.size()];

      j = 0;
      forall_types(it2, nextdata.members) {
        nextargs[j] = tuple_project(inputargs[i], convert_sort(*it2), j);
        j++;
      }

      tuple_create_rec(subname, *it, nextargs);
    } else if (is_tuple_array_ast_type(*it)) {
      // convert_ast will have already, in fact, created a tuple array.
      // We just need to bind it into this one.
      std::string subname = name + data.member_names[i].as_string() + ".";
      const tuple_smt_ast *target =
        new tuple_smt_ast(convert_sort(*it), subname);
      const smt_ast *src = inputargs[i];
      assert_lit(mk_lit(tuple_array_equality(target, src)));
    } else {
      std::string symname = name + data.member_names[i].as_string();
      const smt_sort *sort = convert_sort(*it);
      const smt_ast *args[2];
      args[0] = mk_smt_symbol(symname, sort);
      args[1] = inputargs[i];
      const smt_ast *eq = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(eq);
      assert_lit(l);
    }

    i++;
  }
}

smt_ast *
smt_convt::mk_tuple_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + ".";
  const smt_sort *sort = convert_sort(sym.type);
  return new tuple_smt_ast(sort, name);
}

smt_ast *
smt_convt::mk_tuple_array_symbol(const expr2tc &expr)
{
  const symbol2t &sym = to_symbol2t(expr);
  std::string name = sym.get_symbol_name() + "[]";
  const smt_sort *sort = convert_sort(sym.type);
  return new tuple_smt_ast(sort, name);
}

smt_ast *
smt_convt::tuple_project(const smt_ast *a, const smt_sort *s, unsigned int i)
{
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL && "Non tuple_smt_ast class in smt_convt::tuple_project");

  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(a->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_project");
  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  assert(i < data.members.size() && "Out-of-bounds tuple element accessed");
  const std::string &fieldname = data.member_names[i].as_string();
  std::string sym_name = ta->name + fieldname;

  // Cope with recursive structs.
  const type2tc &restype = data.members[i];
  if (is_tuple_ast_type(restype) || is_tuple_array_ast_type(restype)) {
    sym_name = sym_name + ".";
    return new tuple_smt_ast(s, sym_name);
  } else {
    return mk_smt_symbol(sym_name, s);
  }
}

const smt_ast *
smt_convt::tuple_update(const smt_ast *a, unsigned int i, const smt_ast *v)
{
  // Turn a project into an equality with an update.
  const smt_ast *args[2];
  bvt eqs;
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);

  // Create a fresh tuple to store the result in
  std::string name = mk_fresh_name("tuple_update::");
  const tuple_smt_ast *result = new tuple_smt_ast(a->sort, name);

  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL && "Non tuple_smt_ast class in smt_convt::tuple_update");

  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(ta->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_update");

  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  unsigned int j = 0;
  forall_types(it, data.members) {
    if (j == i) {
      const smt_sort *tmp = convert_sort(*it);
      const smt_ast *thefield = tuple_project(result, tmp, j);
      if (is_tuple_ast_type(*it)) {
        eqs.push_back(mk_lit(tuple_equality(thefield, v)));
      } else {
        args[0] = thefield;
        args[1] = v;
        eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
      }
    } else {
      if (is_tuple_ast_type(*it)) {
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

  assert_lit(land(eqs));
  return result;
}

const smt_ast *
smt_convt::tuple_equality(const smt_ast *a, const smt_ast *b)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  const tuple_smt_ast *tb = dynamic_cast<const tuple_smt_ast *>(b);
  assert(ta != NULL && "Non tuple_smt_ast class in smt_convt::tuple_equality");
  assert(tb != NULL && "Non tuple_smt_ast class in smt_convt::tuple_equality");

  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(ta->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_equality");

  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  std::vector<literalt> lits;
  lits.reserve(data.members.size());

  // Iterate through each field and encode an equality.
  unsigned int i = 0;
  forall_types(it, data.members) {
    if (is_tuple_ast_type(*it)) {
      // Recurse.
      const smt_ast *args[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = tuple_project(a, sort, i);
      args[1] = tuple_project(b, sort, i);
      const smt_ast *eq = tuple_equality(args[0], args[1]);
      literalt l = mk_lit(eq);
      lits.push_back(l);
    } else if (is_tuple_array_ast_type(*it)) {
      const smt_ast *args[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = tuple_project(a, sort, i);
      args[1] = tuple_project(b, sort, i);
      const smt_ast *eq = tuple_array_equality(args[0], args[1]);
      literalt l = mk_lit(eq);
      lits.push_back(l);
    } else {
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

  literalt l = land(lits);
  return lit_to_ast(l);
}

const smt_ast *
smt_convt::tuple_ite(const smt_ast *cond, const smt_ast *true_val,
                     const smt_ast *false_val, const smt_sort *sort)
{
  // Encode as an ite of each element.
  const tuple_smt_ast *trueast = dynamic_cast<const tuple_smt_ast *>(true_val);
  const tuple_smt_ast *falseast = dynamic_cast<const tuple_smt_ast*>(false_val);
  assert(trueast != NULL && "Non tuple_smt_ast class in smt_convt::tuple_ite");
  assert(falseast != NULL && "Non tuple_smt_ast class in smt_convt::tuple_ite");

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
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const tuple_smt_sort *ts =
    dynamic_cast<const tuple_smt_sort *>(true_val->sort);
  assert(ts != NULL && "Non tuple_smt_sort class in smt_convt::tuple_ite");

  const struct_union_data &data =
    dynamic_cast<const struct_union_data &>(*ts->thetype.get());

  // Iterate through each field and encode an ite.
  unsigned int i = 0;
  forall_types(it, data.members) {
    if (is_tuple_ast_type(*it)) {
      // Recurse.
      const tuple_smt_ast *args[3];
      const smt_sort *sort = convert_sort(*it);
      args[0] =
        static_cast<const tuple_smt_ast *>(tuple_project(result, sort, i));
      args[1] =
        static_cast<const tuple_smt_ast *>(tuple_project(true_val, sort, i));
      args[2] =
        static_cast<const tuple_smt_ast *>(tuple_project(false_val, sort, i));
      tuple_ite_rec(args[0], cond, args[1], args[2]);
    } else if (is_tuple_array_ast_type(*it)) {
      // Same deal, but with arrays
      const tuple_smt_ast *args[3];
      const smt_sort *sort = convert_sort(*it);
      args[0] =
        static_cast<const tuple_smt_ast *>(tuple_project(result, sort, i));
      args[1] =
        static_cast<const tuple_smt_ast *>(tuple_project(true_val, sort, i));
      args[2] =
        static_cast<const tuple_smt_ast *>(tuple_project(false_val, sort, i));
      args[1] = static_cast<const tuple_smt_ast*>
          (tuple_array_ite(cond, args[1], args[2], args[1]->sort));
      assert_lit(mk_lit(tuple_array_equality(args[0], args[1])));
    } else {
      const smt_ast *args[3], *eqargs[2];
      const smt_sort *sort = convert_sort(*it);
      args[0] = cond;
      args[1] = tuple_project(true_val, sort, i);
      args[2] = tuple_project(false_val, sort, i);
      eqargs[0] = mk_func_app(sort, SMT_FUNC_ITE, args, 3);
      eqargs[1] = tuple_project(result, sort, i);
      const smt_ast *eq = mk_func_app(boolsort, SMT_FUNC_EQ, eqargs, 2);
      literalt l = mk_lit(eq);
      assert_lit(l);
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
  const smt_sort *sort = convert_sort(array_type);
  std::string name = mk_fresh_name("tuple_array_create::");
  const smt_ast *newsym = new tuple_smt_ast(sort, name);

  // Check size
  const array_type2t &arr_type =
    static_cast<const array_type2t &>(*array_type.get());
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
    // Repeatedly store things into this.
    const smt_ast *init = inputargs[0];
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      newsym = tuple_array_update(newsym, field, init, fieldsort);
    }

    return newsym;
  } else {
    // Repeatedly store things into this.
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      newsym = tuple_array_update(newsym, field, inputargs[i], fieldsort);
    }

    return newsym;
  }
}

const smt_ast *
smt_convt::tuple_array_select(const smt_ast *a, const smt_sort *s,
                              const smt_ast *field)
{
  // Select everything at the given element into a fresh tulple. Don't attempt
  // to support selecting array fields. In the future we can arrange something
  // whereby tuple operations are aware of this array situation and don't
  // have to take this inefficient approach.
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_select");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(a->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_select");

  std::string name = mk_fresh_name("tuple_array_select::");
  const tuple_smt_ast *result = new tuple_smt_ast(s, name);

  const array_type2t &array_type = to_array_type(ts->thetype);
  tuple_array_select_rec(ta, array_type.subtype, result, field);
  return result;
}

void
smt_convt::tuple_array_select_rec(const tuple_smt_ast *ta,
                                  const type2tc &subtype,
                                  const tuple_smt_ast *result,
                                  const smt_ast *field)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = (is_pointer_type(subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*subtype.get());

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
      const smt_sort *sort = convert_sort(*it);
      const tuple_smt_ast *result_field =
        static_cast<const tuple_smt_ast *>(tuple_project(result, sort, i));
      std::string substruct_name =
        ta->name + struct_type.member_names[i].as_string() + ".";
      const tuple_smt_ast *array_name = new tuple_smt_ast(sort, substruct_name);
      tuple_array_select_rec(array_name, *it, result_field, field);
    } else {
      std::string name = ta->name + struct_type.member_names[i].as_string();
      const smt_ast *args[2];
      const smt_sort *field_sort = convert_sort(*it);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, field->sort,field_sort);
      args[0] = mk_smt_symbol(name, arrsort);
      args[1] = field;
      args[0] = mk_func_app(field_sort, SMT_FUNC_SELECT, args, 2);
      args[1] = tuple_project(result, field_sort, i);
      const smt_ast *res = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(res);
      assert_lit(l);
    }

    i++;
  }
}

const smt_ast *
smt_convt::tuple_array_update(const smt_ast *a, const smt_ast *index,
                              const smt_ast *val,
                              const smt_sort *fieldsort __attribute__((unused)))
{
  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_ast *tv = dynamic_cast<const tuple_smt_ast *>(val);
  assert(tv != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(ta->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_update");

  std::string name = mk_fresh_name("tuple_array_select[]::");
  const tuple_smt_ast *result = new tuple_smt_ast(a->sort, name);

  const array_type2t &array_type = to_array_type(ts->thetype);
  tuple_array_update_rec(ta, tv, index, result, array_type.subtype);
  return result;
}

void
smt_convt::tuple_array_update_rec(const tuple_smt_ast *ta,
                                  const tuple_smt_ast *tv, const smt_ast *idx,
                                  const tuple_smt_ast *result,
                                  const type2tc &subtype)
{
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = (is_pointer_type(subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*subtype.get());

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
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

      tuple_array_update_rec(src, val, idx, target, *it);
    } else {
      std::string arrname = ta->name + struct_type.member_names[i].as_string();
      std::string valname = tv->name + struct_type.member_names[i].as_string();
      std::string resname = result->name +
                            struct_type.member_names[i].as_string();
      const smt_ast *args[3];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, idx->sort, idx_sort);
      args[0] = mk_smt_symbol(arrname, arrsort);
      args[1] = idx;
      args[2] = mk_smt_symbol(valname, idx_sort);
      args[0] = mk_func_app(arrsort, SMT_FUNC_STORE, args, 3);
      args[1] = mk_smt_symbol(resname, arrsort);
      const smt_ast *res = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(res);
      assert_lit(l);
    }

    i++;
  }
}

const smt_ast *
smt_convt::tuple_array_equality(const smt_ast *a, const smt_ast *b)
{

  const tuple_smt_ast *ta = dynamic_cast<const tuple_smt_ast *>(a);
  assert(ta != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_equality");
  const tuple_smt_ast *tb = dynamic_cast<const tuple_smt_ast *>(b);
  assert(tb != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_equality");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(a->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_equality");

  const array_type2t &array_type = to_array_type(ts->thetype);
  return tuple_array_equality_rec(ta, tb, array_type.subtype);
}

const smt_ast *
smt_convt::tuple_array_equality_rec(const tuple_smt_ast *a,
                                    const tuple_smt_ast *b,
                                    const type2tc &subtype)
{
  bvt eqs;
  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const struct_union_data &struct_type = (is_pointer_type(subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*subtype.get());

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
      const smt_sort *tmp = convert_sort(*it);
      std::string name1 = a->name + struct_type.member_names[i].as_string()+".";
      std::string name2 = b->name + struct_type.member_names[i].as_string()+".";
      const tuple_smt_ast *new1 = new tuple_smt_ast(tmp, name1);
      const tuple_smt_ast *new2 = new tuple_smt_ast(tmp, name2);
      eqs.push_back(mk_lit(tuple_array_equality_rec(new1, new2, *it)));
    } else {
      std::string name1 = a->name + struct_type.member_names[i].as_string();
      std::string name2 = b->name + struct_type.member_names[i].as_string();
      const smt_ast *args[2];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *dom_sort = machine_int_sort;
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);
      args[0] = mk_smt_symbol(name1, arrsort);
      args[1] = mk_smt_symbol(name2, arrsort);
      eqs.push_back(mk_lit(mk_func_app(boolsort, SMT_FUNC_EQ, args, 2)));
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

  const tuple_smt_ast *tv = dynamic_cast<const tuple_smt_ast *>(trueval);
  assert(tv != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_ast *fv = dynamic_cast<const tuple_smt_ast *>(false_val);
  assert(fv != NULL &&
         "Non tuple_smt_ast class in smt_convt::tuple_array_update");
  const tuple_smt_sort *ts = dynamic_cast<const tuple_smt_sort *>(tv->sort);
  assert(ts != NULL &&
         "Non tuple_smt_sort class in smt_convt::tuple_array_update");

  std::string name = mk_fresh_name("tuple_array_ite[]::");
  const tuple_smt_ast *result = new tuple_smt_ast(tv->sort, name);

  tuple_array_ite_rec(tv, fv, cond, ts->thetype, result);
  return result;
}

void
smt_convt::tuple_array_ite_rec(const tuple_smt_ast *tv, const tuple_smt_ast *fv,
                               const smt_ast *cond, const type2tc &type,
                               const tuple_smt_ast *res)
{

  const smt_sort *boolsort = mk_sort(SMT_SORT_BOOL);
  const array_type2t &array_type = to_array_type(type);
  const struct_union_data &struct_type = (is_pointer_type(array_type.subtype))
    ? *pointer_type_data
    : static_cast<const struct_union_data &>(*array_type.subtype.get());

  unsigned int i = 0;
  forall_types(it, struct_type.members) {
    if (is_tuple_ast_type(*it)) {
      std::cerr << "XXX struct struct array ite unimplemented" << std::endl;
      abort();
    } else {
      std::string tname = tv->name + struct_type.member_names[i].as_string();
      std::string fname = fv->name + struct_type.member_names[i].as_string();
      std::string rname = res->name + struct_type.member_names[i].as_string();
      const smt_ast *args[3];
      const smt_sort *idx_sort = convert_sort(*it);
      const smt_sort *dom_sort = machine_int_sort;
      const smt_sort *arrsort = mk_sort(SMT_SORT_ARRAY, dom_sort, idx_sort);
      args[0] = cond;
      args[1] = mk_smt_symbol(tname, arrsort);
      args[2] = mk_smt_symbol(fname, arrsort);
      args[0] = mk_func_app(idx_sort, SMT_FUNC_ITE, args, 3);
      args[1] = mk_smt_symbol(rname, arrsort);
      args[0] = mk_func_app(boolsort, SMT_FUNC_EQ, args, 2);
      literalt l = mk_lit(args[0]);
      assert_lit(l);
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
  const smt_ast *args[3];
  const smt_sort *sort = convert_sort(expr->type);
  std::string name = mk_fresh_name("array_create::");
  const smt_ast *newsym = mk_smt_symbol(name, sort);

  // Check size
  const array_type2t &arr_type =
    static_cast<const array_type2t &>(*expr->type.get());
  if (arr_type.size_is_infinite) {
    // Guarentee nothing, this is modelling only.
    return newsym;
  } else if (!is_constant_int2t(arr_type.array_size)) {
    std::cerr << "Non-constant sized array of type constant_array_of2t"
              << std::endl;
    abort();
  }

  const constant_int2t &thesize = to_constant_int2t(arr_type.array_size);
  uint64_t sz = thesize.constant_value.to_ulong();

  if (is_constant_array_of2t(expr)) {
    const constant_array_of2t &array = to_constant_array_of2t(expr);

    // Repeatedly store things into this.
    const smt_ast *init = convert_ast(array.initializer);
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      args[0] = newsym;
      args[1] = field;
      args[2] = init;
      newsym = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
    }

    return newsym;
  } else {
    assert(is_constant_array2t(expr));
    const constant_array2t &array = to_constant_array2t(expr);

    // Repeatedly store things into this.
    for (unsigned int i = 0; i < sz; i++) {
      const smt_ast *field = (int_encoding)
        ? mk_smt_int(BigInt(i), false)
        : mk_smt_bvint(BigInt(i), false, config.ansi_c.int_width);
      args[0] = newsym;
      args[1] = field;
      args[2] = convert_ast(array.datatype_members[i]);
      newsym = mk_func_app(sort, SMT_FUNC_STORE, args, 3);
    }

    return newsym;
  }
}

const smt_ast *
smt_convt::tuple_array_create_despatch(const expr2tc &expr,
                                       const smt_sort *domain)
{

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
