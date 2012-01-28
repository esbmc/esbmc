/*******************************************************************\

   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <iostream>
#include <iomanip>
#include <math.h>
#include <iomanip>
#include <sstream>
#include <string>

#include <arith_tools.h>
#include <std_expr.h>
#include <solvers/flattening/boolbv_type.h>
#include <solvers/flattening/boolbv_width.h>
#include <solvers/flattening/boolbv.h>

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

  if (expr.id() == exprt::symbol ||
      expr.id() == "nondet_symbol") {
    std::string identifier, tmp;
    Z3_ast bv;

    identifier = expr.identifier().as_string();

    map_varst::const_iterator cache_result = map_vars.find(identifier.c_str());
    if (cache_result != map_vars.end()) {
      bv = cache_result->second;
      return bv_get_rec(bv, expr.type());
    } else {
      std::cerr << "Unrecognized symbol in z3_get: " << identifier << std::endl;
      abort();
    }
  } else if (expr.id() == exprt::constant) {
    return expr;
  } else {
    std::cerr << "Unrecognized irep fetched from Z3: " << expr.id().as_string();
    std::cerr << std::endl;
    abort();
  }
}

exprt
z3_convt::bv_get_rec(const Z3_ast bv, const typet &type) const
{
  Z3_ast tmp;
  Z3_app app;
  unsigned width;

  app = Z3_to_app(z3_ctx, bv); // Just typecasting.

  get_type_width(type, width);

  if (type.is_bool()) {
    Z3_app app = Z3_to_app(z3_ctx, bv);
    if (Z3_get_bool_value(z3_ctx, Z3_app_to_ast(z3_ctx, app)) == Z3_L_TRUE)
      return true_exprt();
    else
      return false_exprt();
  } else if (type.is_array()) {
    std::vector<exprt> unknown;
    exprt expr;

    unsigned num_fields = Z3_get_app_num_args(z3_ctx, Z3_to_app(z3_ctx, bv));
    unknown.resize(num_fields);

    for (unsigned int i = 0; i < num_fields; i++) {
      tmp = Z3_get_app_arg(z3_ctx, app, i);
      unknown.push_back(bv_get_rec(tmp, type.subtype()));
    }

    exprt dest = exprt("array", type);
    dest.operands() = unknown;
    return dest;
  } else if (type.id() == "struct") {
    std::vector<exprt> unknown;
    const irept &components = type.components();
    exprt::operandst op;
    op.reserve(components.get_sub().size());

    exprt expr;
    unsigned i = 0;
    Z3_app app = Z3_to_app(z3_ctx, bv);
    unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
    Z3_ast tmp;

    if (num_fields == 0)
      return nil_exprt();


    forall_irep(it, components.get_sub())
    {
      const typet &subtype = it->type();
      op.push_back(nil_exprt());
      if (subtype.id() != "pointer") { //@TODO: beautify counter-examples that
                                       // contain pointers
        tmp = Z3_get_app_arg(z3_ctx, app, i);
        expr = bv_get_rec(tmp, subtype);
        if (!expr.is_nil())
          unknown.push_back(expr);
        else
          return nil_exprt();

        op.back() = unknown.back();

        ++i;
    }
    }

    exprt dest = exprt(type.id(), type);
    dest.operands().swap(op);
    return dest;
  } else if (type.id() == "union") {
    std::vector<exprt> unknown;
    const irept &components = type.components();

    unsigned component_nr = 0;

    if (component_nr >= components.get_sub().size())
      return nil_exprt();

    exprt value("union", type);
    value.operands().resize(1);

    value.component_name(components.get_sub()[component_nr].name());

    exprt::operandst op;
    op.reserve(1);

    exprt expr;
    unsigned i = 0;
    Z3_app app = Z3_to_app(z3_ctx, bv);
    unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
    Z3_ast tmp;

    tmp = Z3_get_app_arg(z3_ctx, app, num_fields - 1);
    std::string value1;

    if (Z3_get_ast_kind(z3_ctx, tmp) == Z3_NUMERAL_AST)
      value1 = Z3_get_numeral_string(z3_ctx, tmp);

    unsigned int comp_nr = atoi(value1.c_str());

    if (num_fields == 0)
      return nil_exprt();

    forall_irep(it, components.get_sub())
    {
      const typet &subtype = it->type();

      if (subtype.id() != "pointer") { //@TODO
        tmp = Z3_get_app_arg(z3_ctx, app, i);
        expr = bv_get_rec(tmp, subtype);
        if (comp_nr == i) {
          if (!expr.is_nil())
            unknown.push_back(expr);
          op.push_back(unknown.back());
          break;
        }
        ++i;
      }
    }

    value.operands().swap(op);

    return value;
  } else if (type.id() == "pointer") {
    exprt object, offset;
    Z3_app app = Z3_to_app(z3_ctx, bv);
    unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
    Z3_ast tmp;

    assert(num_fields == 2);

    const typet &subtype = static_cast<const typet &>(type.subtype());

    tmp = Z3_get_app_arg(z3_ctx, app, 0); //object
    object = bv_get_rec(tmp, unsignedbv_typet(config.ansi_c.int_width));
    tmp = Z3_get_app_arg(z3_ctx, app, 1); //offset
    offset = bv_get_rec(tmp, unsignedbv_typet(config.ansi_c.int_width));

    pointer_logict::pointert pointer;
    pointer.object =
      integer2long(binary2integer(object.value().as_string(), false));
    pointer.offset = binary2integer(offset.value().as_string(), true);
    if (pointer.object == 0) {
      constant_exprt result(type);
      result.set_value("NULL");
      return result;
    }

    return pointer_logic.pointer_expr(pointer, type);
  } else if (type.id() == "signedbv" || type.id() == "unsignedbv") {
    unsigned width;
    if (Z3_get_ast_kind(z3_ctx, bv) != Z3_NUMERAL_AST)
      return nil_exprt();
    std::string value = Z3_get_numeral_string(z3_ctx, bv);
    boolbv_get_width(type, width);
    constant_exprt value_expr(type);
    value_expr.set_value(integer2binary(string2integer(value), width));
    return value_expr;
  } else if (type.id() == "fixedbv" && int_encoding) {
    if (Z3_get_ast_kind(z3_ctx, bv) != Z3_NUMERAL_AST)
      return nil_exprt();
    std::string value = Z3_get_numeral_string(z3_ctx, bv);
    get_type_width(type, width);
    constant_exprt value_expr(type);
    value_expr.set_value(get_fixed_point(width, value));
    return value_expr;
  } else if (type.id() == "c_enum") {
    if (Z3_get_ast_kind(z3_ctx, bv) != Z3_NUMERAL_AST)
      return nil_exprt();
    std::string value = Z3_get_numeral_string(z3_ctx, bv);
    constant_exprt value_expr(type);
    value_expr.set_value(value);
    return value_expr;
  } else {
//    std::cerr << "Unrecognized type \"" << type.id() << "\" generating counterexample" << std::endl;
//    abort();
    return nil_exprt();
  }
}
