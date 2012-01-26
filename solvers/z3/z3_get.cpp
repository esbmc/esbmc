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

  if ((expr.type().is_array() && expr.type().subtype().is_array()) ||
      (expr.type().is_array() && expr.type().subtype().id() == "pointer") ||
      (expr.type().is_array() && expr.type().subtype().id() == "union") ||
      (expr.type().is_array() && expr.type().subtype().id() == "struct"))
    return expr;

  if (expr.id() == exprt::symbol ||
      expr.id() == "nondet_symbol") {
    std::string identifier, tmp;
    std::vector<exprt> unknown;
    Z3_ast bv;

    identifier = expr.identifier().as_string();

    if (expr.type().id() == "pointer") {
      for (z3_cachet::const_iterator it = z3_cache.begin();
           it != z3_cache.end(); it++)
      {
	if (identifier.compare((*it).second.c_str()) == 0) {
	  if ((*it).first.id() == exprt::symbol || (*it).first.id() ==
	      exprt::constant)
	    return (*it).first;
	  else
	    return nil_exprt();
	}
      }
      return nil_exprt();
    }

    map_varst::const_iterator cache_result = map_vars.find(identifier.c_str());
    if (cache_result != map_vars.end()) {
      bv = cache_result->second;
      return bv_get_rec(bv, unknown, true, expr.type());
    }

    return bv_get_rec(bv, unknown, false, expr.type());
  } else if (expr.id() == exprt::constant)
    return expr;

  return expr;
}

void
z3_convt::fill_vector(
  const Z3_ast &bv, std::vector<exprt> &unknown, const typet &type) const
{

  unsigned i, width;
  static unsigned int idx;
  Z3_app app = Z3_to_app(z3_ctx, bv);
  unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
  Z3_ast tmp;
  std::string value;

  for (i = 0; i < num_fields; i++)
  {
    tmp = Z3_get_app_arg(z3_ctx, app, i);

    if (Z3_get_ast_kind(z3_ctx, tmp) == Z3_NUMERAL_AST) {
      boolbv_get_width(type, width);
      value = Z3_get_numeral_string(z3_ctx, tmp);
      constant_exprt value_expr(type);
      value_expr.set_value(integer2binary(string2integer(value), width));

      if (i == 1)
	idx = atoi(value.c_str());
      else if (i == 2)
	if (idx < unknown.size())
	  unknown[idx] = value_expr;
    } else   {
      fill_vector(tmp, unknown, type);
    }
  }
}

exprt
z3_convt::bv_get_rec(
  const Z3_ast &bv, std::vector<exprt> &unknown, const bool cache,
  const typet &type) const
{

  unsigned width;

  if (boolbv_get_width(type, width))
    return nil_exprt();

  if (type.is_bool()) {
    std::string value;
    size_t found;

    if (cache) {
      Z3_app app = Z3_to_app(z3_ctx, bv);
      Z3_func_decl d = Z3_get_app_decl(z3_ctx, app);
      value = Z3_func_decl_to_string(z3_ctx, d);
      found = value.find("true");
    } else   {
      found = std::string::npos;
    }

    if (found != std::string::npos)
      return true_exprt();
    else
      return false_exprt();  // default
  }

  bvtypet bvtype = get_bvtype(type);

  if (bvtype == IS_UNKNOWN) {
    if (type.is_array()) {
      unsigned sub_width;
      const typet &subtype = type.subtype();

      if (subtype.id() == "struct") //@TODO
	return nil_exprt();

      if (!boolbv_get_width(subtype, sub_width)) {
	exprt expr;
	static exprt::operandst op;
	constant_exprt zero_expr(subtype);

	op.reserve(width / sub_width);
	unsigned num_fields = Z3_get_app_num_args(z3_ctx, Z3_to_app(z3_ctx, bv));

	if (num_fields == 0)
	  return nil_exprt();

	unknown.resize(width / sub_width);

	fill_vector(bv, unknown, subtype);

	if (unknown.size() == 0)
	  return nil_exprt();

	unsigned int size = unknown.size();
	zero_expr.set_value("0");

	for (unsigned i = 0; i < size; i++)
	{
	  expr = unknown[i];

	  if (expr.value().as_string().compare("") == 0)
	    op.push_back(zero_expr);
	  else
	    op.push_back(expr);
	}

	if (op.empty())
	  return nil_exprt();

	exprt dest = exprt("array", type);
	dest.operands().swap(op);
	return dest;
      }
    } else if (type.id() == "struct") {
      const irept &components = type.components();
      exprt::operandst op;
      op.reserve(components.get_sub().size());
      unsigned int size;

      size = components.get_sub().size();

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
	  unsigned sub_width;

	  if (!boolbv_get_width(subtype, sub_width)) {
	    tmp = Z3_get_app_arg(z3_ctx, app, i);
	    expr = bv_get_rec(tmp, unknown, true, subtype);
	    if (!expr.is_nil())
	      unknown.push_back(expr);
	    else
	      return nil_exprt();

	    op.back() = unknown.back();

	    ++i;
	  }
	}
      }

      exprt dest = exprt(type.id(), type);
      dest.operands().swap(op);
      return dest;
    } else if (type.id() == "union")      {
      const irept &components = type.components();

      unsigned component_nr = 0;

      if (component_nr >= components.get_sub().size())
	return nil_exprt();

      exprt value("union", type);
      value.operands().resize(1);

      value.component_name(components.get_sub()[component_nr].name());

      exprt::operandst op;
      op.reserve(1);
      unsigned int size;

      size = components.get_sub().size() + 1;

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
	  unsigned sub_width;
	  if (!boolbv_get_width(subtype, sub_width)) {
	    tmp = Z3_get_app_arg(z3_ctx, app, i);
	    expr = bv_get_rec(tmp, unknown, true, subtype);
	    if (comp_nr == i) {
	      if (!expr.is_nil())
		unknown.push_back(expr);
	      op.push_back(unknown.back());
	      break;
	    }
	    ++i;
	  }
	}
      }

      value.operands().swap(op);

      return value;
    }
#if 0
    else if (type.id() == "pointer") {
      exprt object, offset;
      Z3_app app = Z3_to_app(z3_ctx, bv);
      unsigned num_fields = Z3_get_app_num_args(z3_ctx, app);
      Z3_ast tmp;

      assert(num_fields == 2);

      const typet &subtype = static_cast<const typet &>(type.subtype());

      tmp = Z3_get_app_arg(z3_ctx, app, 0); //object
      object = bv_get_rec(tmp, unknown, true, subtype);
      tmp = Z3_get_app_arg(z3_ctx, app, 1); //offset
      offset = bv_get_rec(tmp, unknown, true, subtype);

      pointer_logict::pointert pointer;
      pointer.object =
        integer2long(binary2integer(object.value().as_string(), false));
      pointer.offset = binary2integer(offset.value().as_string(), true);
      if (pointer.offset < 0) {
	constant_exprt result(type);
	result.set_value("NULL");
	return result;
      }

      return pointer_logic.pointer_expr(pointer, type);
    }
#endif
  }

  std::string value;
  Z3_ast_kind ast_kind;

  ast_kind = Z3_get_ast_kind(z3_ctx, bv);

  if (ast_kind == Z3_NUMERAL_AST)
    value = Z3_get_numeral_string(z3_ctx, bv);
  else
    return nil_exprt();

  if (type.id() == "fixedbv" && int_encoding) {
    if (boolbv_get_width(type, width))
      return nil_exprt();

    constant_exprt value_expr(type);
    value_expr.set_value(get_fixed_point(width, value));
    return value_expr;
  }

  switch (bvtype) {
  case IS_C_ENUM:
  {
    constant_exprt value_expr(type);
    value_expr.set_value(value);
    return value_expr;
  }
  break;

  case IS_UNKNOWN:
    break;

  case IS_RANGE:
  {
    mp_integer int_value = string2integer(value, false);
    mp_integer from = string2integer(type.from().as_string());

    constant_exprt value_expr(type);
    value_expr.set_value(integer2string(int_value + from));
    return value_expr;
  }
  break;

  default:
    unsigned width;
    boolbv_get_width(type, width);
    constant_exprt value_expr(type);
    value_expr.set_value(integer2binary(string2integer(value), width));
    return value_expr;
  }

  return nil_exprt();
}
