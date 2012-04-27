/*******************************************************************\

   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <migrate.h>
#include <iomanip>
#include <sstream>
#include <string>

#include <arith_tools.h>
#include <std_expr.h>
#include <std_types.h>

#include "z3_conv.h"

std::string
z3_convt::double2string(double d) const
{

  std::ostringstream format_message;
  format_message << std::setprecision(12) << d;
  return format_message.str();
}

std::string
z3_convt::get_fixed_point(const unsigned width, std::string value) const
{

  std::string m, f, tmp;
  size_t found, size;
  double v, magnitude, fraction, expoent;

  found = value.find_first_of("/");
  size = value.size();
  m = value.substr(0, found);
  f = value.substr(found + 1, size);

  v = atof(m.c_str()) / atof(f.c_str());
  magnitude = (int)v;
  fraction = v - magnitude;
  tmp = integer2string(power(2, width / 2), 10);
  expoent = atof(tmp.c_str());
  fraction = fraction * expoent;
  fraction = floor(fraction);
  value = integer2binary(string2integer(double2string(magnitude), 10),width / 2)
                         +integer2binary(string2integer(double2string(fraction),
                         10), width / 2);

  return value;
}

exprt
z3_convt::get(const exprt &expr) const
{

  expr2tc new_expr;
  migrate_expr(expr, new_expr);

  try {

  if (is_symbol2t(new_expr)) {
    std::string identifier, tmp;
    Z3_sort sort;
    Z3_ast bv;
    Z3_func_decl func;

    const symbol2t sym = to_symbol2t(new_expr);
    identifier = sym.name.as_string();
    new_expr->type->convert_smt_type(*this, (void*&)sort);
    bv = z3_api.mk_var(identifier.c_str(), sort);
    func = Z3_get_app_decl(z3_ctx, Z3_to_app(z3_ctx, bv));

    if(Z3_eval_func_decl(z3_ctx, model, func, &bv) == Z3_L_FALSE) {
      // This symbol doesn't have an assignment in this model
      return nil_exprt();
    }

    expr2tc ret = bv_get_rec(bv, new_expr->type);
    return migrate_expr_back(ret);
  } else if (is_constant_expr(new_expr)) {
    return expr;
  } else {
    std::cerr << "Unrecognized irep fetched from Z3: " << expr.id().as_string();
    std::cerr << std::endl;
    abort();
  }

  } catch (conv_error *e) {
    std::cerr << "Conversion error fetching counterexample:" << std::endl;
    std::cerr << e->to_string() << std::endl;
    return nil_exprt();
  }
}

expr2tc
z3_convt::bv_get_rec(const Z3_ast bv, const type2tc &type) const
{
  Z3_app app;
  unsigned width;

  app = Z3_to_app(z3_ctx, bv); // Just typecasting.

  try {
    width = type->get_width();
  } catch (array_type2t::inf_sized_array_excp *) {
    // Not a problem, we don't use the array size in extraction
    width = 0;
  } catch (array_type2t::dyn_sized_array_excp *e) {
    // Also fine.
    width = 0;
  }

  if (is_bool_type(type)) {
    if (Z3_get_bool_value(z3_ctx, Z3_app_to_ast(z3_ctx, app)) == Z3_L_TRUE)
      return expr2tc(new constant_bool2t(true));
    else
      return expr2tc(new constant_bool2t(false));
  } else if (is_array_type(type)) {
    typedef std::pair<mp_integer, expr2tc> array_elem;
    const array_type2t & type_ref = to_array_type(type);
    std::list<array_elem> elems_in_z3_order;
    std::map<mp_integer, expr2tc> mapped_elems;
    exprt expr;

    // Array model is a series of store ASTs, with the operands:
    //   0) Array to store into
    //   1) Index
    //   2) Value
    // As with SMT everything, the array to store into is another store
    // instruction, so we need to recurse into it. Fetch all these pieces of data
    // out, store in a list, descend through stores.
    Z3_app recurse_store = Z3_to_app(z3_ctx, bv);
    while (Z3_get_app_num_args(z3_ctx, recurse_store) == 3) {
      Z3_ast idx, value;
      idx = Z3_get_app_arg(z3_ctx, recurse_store, 1);
      value = Z3_get_app_arg(z3_ctx, recurse_store, 2);
      recurse_store = Z3_to_app(z3_ctx, Z3_get_app_arg(z3_ctx, recurse_store, 0));

      assert(Z3_get_ast_kind(z3_ctx, idx) == Z3_NUMERAL_AST);
      std::string index = Z3_get_numeral_string(z3_ctx, idx);
      mp_integer i = string2integer(index);
      expr2tc val = bv_get_rec(value, type_ref.subtype);

      elems_in_z3_order.push_back(array_elem(i, val));
    }

    // We now have all assignments to the array; including to duplicate indexes.
    // So, put everything into a map in reverse order from how we received it,
    // ensuring that the assignment to a particular index is the most recent.
    for (std::list<array_elem>::reverse_iterator it = elems_in_z3_order.rbegin();
         it != elems_in_z3_order.rend(); it++)
      mapped_elems[it->first] = it->second;

    // Finally, serialise into operands list

    std::vector<expr2tc> elem_list;
    for (std::map<mp_integer, expr2tc>::const_iterator it =mapped_elems.begin();
         it != mapped_elems.end(); it++)
      elem_list.push_back(it->second);

    // XXXjmorse - this isn't going to be printed right if the array data is
    // sparse. See trac #73

    return expr2tc(new constant_array2t(type, elem_list));
  } else if (is_struct_type(type)) {
    const struct_type2t &type_ref = to_struct_type(type);
    std::vector<expr2tc> unknown;
    std::vector<expr2tc> opers;
    opers.reserve(type_ref.members.size());

    expr2tc expr;
    unsigned i = 0;
    unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
    Z3_ast tmp;

    if (num_fields == 0)
      return expr2tc();

    forall_types(it, type_ref.members) {
      tmp = Z3_get_app_arg(z3_ctx, app, i++);
      opers.push_back(bv_get_rec(tmp, *it));
    }

    return expr2tc(new constant_struct2t(type, opers));
  } else if (is_union_type(type)) {
    const union_type2t &type_ref = to_union_type(type);
    unsigned component_nr = 0;
    std::vector<expr2tc> operands;

    if (component_nr >= type_ref.members.size())
      return expr2tc();

    expr2tc expr;
    unsigned int i = 0;
    int comp_nr;
    unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
    Z3_ast tmp;

    tmp = Z3_get_app_arg(z3_ctx, app, num_fields - 1);

    assert(Z3_get_ast_kind(z3_ctx, tmp) == Z3_NUMERAL_AST);
    Z3_bool tbool = Z3_get_numeral_int(z3_ctx, tmp, &comp_nr);
    assert(tbool);

    if (num_fields == 0)
      return expr2tc();

    forall_types(it, type_ref.members)
    {
      tmp = Z3_get_app_arg(z3_ctx, app, i);
      expr = bv_get_rec(tmp, *it);
      operands.push_back(expr);
      if (comp_nr == i) {
        // XXXjmorse, Dunno what to do with this
        // in fact, shouldn't be reached, not in components list.
        break;
      }
      ++i;
    }

    return expr2tc(new constant_union2t(type, operands));
  } else if (is_pointer_type(type)) {
    expr2tc object, offset;
    unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
    Z3_ast tmp;

    if (num_fields != 2) {
      std::cerr << "pointer symbol retrieval error" << std::endl;
      return expr2tc();
    }

    assert(num_fields == 2);

    tmp = Z3_get_app_arg(z3_ctx, app, 0); //object
    object = bv_get_rec(tmp, type_pool.get_uint(config.ansi_c.int_width));
    tmp = Z3_get_app_arg(z3_ctx, app, 1); //offset
    offset = bv_get_rec(tmp, type_pool.get_uint(config.ansi_c.int_width));

    assert(is_unsignedbv_type(object->type));
// XXXjmorse - some thought should go in here.
//    assert(is_signedbv_type(offset->type));
    const constant_int2t &objref = to_constant_int2t(object);
    const constant_int2t &offsref = to_constant_int2t(offset);

    pointer_logict::pointert pointer;
    pointer.object = objref.constant_value.to_ulong();
    pointer.offset = offsref.constant_value;
    if (pointer.object == 0) {
      return expr2tc(new symbol2t(type, "NULL"));
    }

    return pointer_logic.pointer_expr(pointer, type);
  } else if (is_bv_type(type)) {
    if (Z3_get_ast_kind(z3_ctx, bv) != Z3_NUMERAL_AST)
      return expr2tc();
    std::string value = Z3_get_numeral_string(z3_ctx, bv);
    return expr2tc(new constant_int2t(type, BigInt(value.c_str())));
  } else if (is_fixedbv_type(type) && int_encoding) {
    if (Z3_get_ast_kind(z3_ctx, bv) != Z3_NUMERAL_AST)
      return expr2tc();
    std::string value = Z3_get_numeral_string(z3_ctx, bv);
    constant_exprt value_expr(migrate_type_back(type));
    value_expr.set_value(get_fixed_point(width, value));
    fixedbvt fbv;
    fbv.from_expr(value_expr);
    return expr2tc(new constant_fixedbv2t(type, fbv));
  } else if (is_fixedbv_type(type) && !int_encoding) {
    // bv integer representation of fixedbv can be stuffed right back into a
    // constant irep, afaik
    if (Z3_get_ast_kind(z3_ctx, bv) != Z3_NUMERAL_AST)
      return expr2tc();
    std::string value = Z3_get_numeral_string(z3_ctx, bv);
    constant_exprt value_expr(migrate_type_back(type));
    value_expr.set_value(integer2binary(string2integer(value), width));
    fixedbvt fbv;
    fbv.from_expr(value_expr);
    return expr2tc(new constant_fixedbv2t(type, fbv));
  } else {
    std::cerr << "Unrecognized type \"" << type->type_names[type->type_id] <<
                 "\" generating counterexample" << std::endl;
    abort();
  }
}
