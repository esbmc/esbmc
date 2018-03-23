/*******************************************************************\

Module: Expression Representation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cstdlib>
#include <util/expr.h>
#include <util/fixedbv.h>
#include <util/ieee_float.h>
#include <util/mp_arith.h>

void exprt::move_to_operands(exprt &expr)
{
  operandst &op = operands();
  op.push_back(static_cast<const exprt &>(get_nil_irep()));
  op.back().swap(expr);
}

void exprt::move_to_operands(exprt &e1, exprt &e2)
{
  operandst &op = operands();
#ifndef USE_LIST
  op.reserve(op.size() + 2);
#endif
  op.push_back(static_cast<const exprt &>(get_nil_irep()));
  op.back().swap(e1);
  op.push_back(static_cast<const exprt &>(get_nil_irep()));
  op.back().swap(e2);
}

void exprt::move_to_operands(exprt &e1, exprt &e2, exprt &e3)
{
  operandst &op = operands();
#ifndef USE_LIST
  op.reserve(op.size() + 3);
#endif
  op.push_back(static_cast<const exprt &>(get_nil_irep()));
  op.back().swap(e1);
  op.push_back(static_cast<const exprt &>(get_nil_irep()));
  op.back().swap(e2);
  op.push_back(static_cast<const exprt &>(get_nil_irep()));
  op.back().swap(e3);
}

void exprt::copy_to_operands(const exprt &expr)
{
  operands().push_back(expr);
}

void exprt::copy_to_operands(const exprt &e1, const exprt &e2)
{
  operandst &op = operands();
#ifndef USE_LIST
  op.reserve(op.size() + 2);
#endif
  op.push_back(e1);
  op.push_back(e2);
}

void exprt::copy_to_operands(const exprt &e1, const exprt &e2, const exprt &e3)
{
  operandst &op = operands();
#ifndef USE_LIST
  op.reserve(op.size() + 3);
#endif
  op.push_back(e1);
  op.push_back(e2);
  op.push_back(e3);
}

void exprt::make_typecast(const typet &_type)
{
  exprt new_expr(exprt::typecast);

  new_expr.move_to_operands(*this);
  new_expr.set(i_type, _type);

  swap(new_expr);
}

void exprt::make_not()
{
  if(is_true())
  {
    make_false();
    return;
  }
  if(is_false())
  {
    make_true();
    return;
  }

  exprt new_expr;

  if(id() == i_not && operands().size() == 1)
  {
    new_expr.swap(operands().front());
  }
  else
  {
    new_expr = exprt(i_not, type());
    new_expr.move_to_operands(*this);
  }

  swap(new_expr);
}

bool exprt::is_constant() const
{
  return id() == constant;
}

bool exprt::is_true() const
{
  return is_constant() && type().is_bool() && get(a_value) != "false";
}

bool exprt::is_false() const
{
  return is_constant() && type().is_bool() && get(a_value) == "false";
}

void exprt::make_bool(bool value)
{
  *this = exprt(constant, typet("bool"));
  set(a_value, value ? i_true : i_false);
}

void exprt::make_true()
{
  *this = exprt(constant, typet("bool"));
  set(a_value, i_true);
}

void exprt::make_false()
{
  *this = exprt(constant, typet("bool"));
  set(a_value, i_false);
}

bool operator<(const exprt &X, const exprt &Y)
{
  return (irept &)X < (irept &)Y;
}

void exprt::negate()
{
  const irep_idt &type_id = type().id();

  if(type_id == "bool")
    make_not();
  else
    make_nil();
}

bool exprt::is_boolean() const
{
  return type().is_bool();
}

bool exprt::is_zero() const
{
  if(is_constant())
  {
    const std::string &value = get_string(a_value);
    const irep_idt &type_id = type().id_string();

    if(type_id == "unsignedbv" || type_id == "signedbv")
    {
      mp_integer int_value = binary2integer(value, false);
      if(int_value == 0)
        return true;
    }
    else if(type_id == "fixedbv")
    {
      if(fixedbvt(to_constant_expr(*this)) == 0)
        return true;
    }
    else if(type_id == "floatbv")
    {
      if(ieee_floatt(to_constant_expr(*this)) == 0)
        return true;
    }
    else if(type_id == "pointer")
    {
      if(value == "NULL")
        return true;
    }
  }

  return false;
}

bool exprt::is_one() const
{
  if(is_constant())
  {
    const std::string &value = get_string(a_value);
    const irep_idt &type_id = type().id_string();

    if(type_id == "unsignedbv" || type_id == "signedbv")
    {
      mp_integer int_value = binary2integer(value, false);
      if(int_value == 1)
        return true;
    }
    else if(type_id == "fixedbv")
    {
      if(fixedbvt(to_constant_expr(*this)) == 1)
        return true;
    }
    else if(type_id == "floatbv")
    {
      if(ieee_floatt(to_constant_expr(*this)) == 1)
        return true;
    }
  }

  return false;
}

bool exprt::sum(const exprt &expr)
{
  if(!is_constant() || !expr.is_constant())
    return true;
  if(type() != expr.type())
    return true;

  const irep_idt &type_id = type().id();

  if(type_id == "unsignedbv" || type_id == "signedbv")
  {
    set(
      a_value,
      integer2binary(
        binary2integer(get_string(a_value), false) +
          binary2integer(expr.get_string(a_value), false),
        atoi(type().width().c_str())));
    return false;
  }
  if(type_id == "fixedbv")
  {
    fixedbvt f(to_constant_expr(*this));
    f += fixedbvt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }
  else if(type_id == "floatbv")
  {
    ieee_floatt f(to_constant_expr(*this));
    f += ieee_floatt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }

  return true;
}

bool exprt::mul(const exprt &expr)
{
  if(!is_constant() || !expr.is_constant())
    return true;
  if(type() != expr.type())
    return true;

  const irep_idt &type_id = type().id();

  if(type_id == "unsignedbv" || type_id == "signedbv")
  {
    set(
      a_value,
      integer2binary(
        binary2integer(get_string(a_value), false) *
          binary2integer(expr.get_string(a_value), false),
        atoi(type().width().c_str())));
    return false;
  }
  if(type_id == "fixedbv")
  {
    fixedbvt f(to_constant_expr(*this));
    f *= fixedbvt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }
  else if(type_id == "floatbv")
  {
    ieee_floatt f(to_constant_expr(*this));
    f *= ieee_floatt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }

  return true;
}

bool exprt::subtract(const exprt &expr)
{
  if(!is_constant() || !expr.is_constant())
    return true;

  if(type() != expr.type())
    return true;

  const irep_idt &type_id = type().id();

  if(type_id == "unsignedbv" || type_id == "signedbv")
  {
    set(
      a_value,
      integer2binary(
        binary2integer(get_string(a_value), false) -
          binary2integer(expr.get_string(a_value), false),
        atoi(type().width().c_str())));
    return false;
  }
  if(type_id == "fixedbv")
  {
    fixedbvt f(to_constant_expr(*this));
    f -= fixedbvt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }
  else if(type_id == "floatbv")
  {
    ieee_floatt f(to_constant_expr(*this));
    f -= ieee_floatt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }

  return true;
}

const locationt &exprt::find_location() const
{
  const locationt &l = location();

  if(l.is_not_nil())
    return l;

  forall_operands(it, (*this))
  {
    const locationt &l = it->find_location();
    if(l.is_not_nil())
      return l;
  }

  return static_cast<const locationt &>(get_nil_irep());
}

irep_idt exprt::trans = dstring("trans");
irep_idt exprt::symbol = dstring("symbol");
irep_idt exprt::plus = dstring("+");
irep_idt exprt::minus = dstring("-");
irep_idt exprt::mult = dstring("*");
irep_idt exprt::div = dstring("/");
irep_idt exprt::mod = dstring("mod");
irep_idt exprt::equality = dstring("=");
irep_idt exprt::notequal = dstring("notequal");
irep_idt exprt::index = dstring("index");
irep_idt exprt::arrayof = dstring("array_of");
irep_idt exprt::objdesc = dstring("object_descriptor");
irep_idt exprt::dynobj = dstring("dynamic_object");
irep_idt exprt::typecast = dstring("typecast");
irep_idt exprt::implies = dstring("=>");
irep_idt exprt::i_and = dstring("and");
irep_idt exprt::i_xor = dstring("xor");
irep_idt exprt::i_or = dstring("or");
irep_idt exprt::i_not = dstring("not");
irep_idt exprt::addrof = dstring("address_of");
irep_idt exprt::deref = dstring("dereference");
irep_idt exprt::i_if = dstring("if");
irep_idt exprt::with = dstring("with");
irep_idt exprt::member = dstring("member");
irep_idt exprt::isnan = dstring("isnan");
irep_idt exprt::ieee_floateq = dstring("ieee_float_equal");
irep_idt exprt::i_type = dstring("type");
irep_idt exprt::constant = dstring("constant");
irep_idt exprt::i_true = dstring("true");
irep_idt exprt::i_false = dstring("false");
irep_idt exprt::i_lt = dstring("<");
irep_idt exprt::i_gt = dstring(">");
irep_idt exprt::i_le = dstring("<=");
irep_idt exprt::i_ge = dstring(">=");
irep_idt exprt::i_bitand = dstring("bitand");
irep_idt exprt::i_bitor = dstring("bitor");
irep_idt exprt::i_bitxor = dstring("bitxor");
irep_idt exprt::i_bitnand = dstring("bitnand");
irep_idt exprt::i_bitnor = dstring("bitnor");
irep_idt exprt::i_bitnxor = dstring("bitnxor");
irep_idt exprt::i_bitnot = dstring("bitnot");
irep_idt exprt::i_ashr = dstring("ashr");
irep_idt exprt::i_lshr = dstring("lshr");
irep_idt exprt::i_shl = dstring("shl");
irep_idt exprt::abs = dstring("abs");
irep_idt exprt::argument = dstring("argument");

irep_idt exprt::a_value = dstring("value");

irep_idt exprt::o_operands = dstring("operands");
irep_idt exprt::o_location = dstring("#location");
