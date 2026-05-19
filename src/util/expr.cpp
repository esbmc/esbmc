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
  if (is_true())
  {
    make_false();
    return;
  }
  if (is_false())
  {
    make_true();
    return;
  }

  exprt new_expr;

  if (id() == i_not && operands().size() == 1)
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

  if (type_id == "bool")
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
  if (is_constant())
  {
    const std::string &value = get_string(a_value);
    const irep_idt &type_id = type().id_string();

    if (type_id == "unsignedbv" || type_id == "signedbv")
    {
      BigInt int_value = binary2integer(value, false);
      if (int_value == 0)
        return true;
    }
    else if (type_id == "fixedbv")
    {
      if (fixedbvt(to_constant_expr(*this)) == 0)
        return true;
    }
    else if (type_id == "floatbv")
    {
      if (ieee_floatt(to_constant_expr(*this)) == 0)
        return true;
    }
    else if (type_id == "pointer")
    {
      if (value == "NULL")
        return true;
    }
  }

  return false;
}

bool exprt::is_one() const
{
  if (is_constant())
  {
    const std::string &value = get_string(a_value);
    const irep_idt &type_id = type().id_string();

    if (type_id == "unsignedbv" || type_id == "signedbv")
    {
      BigInt int_value = binary2integer(value, false);
      if (int_value == 1)
        return true;
    }
    else if (type_id == "fixedbv")
    {
      if (fixedbvt(to_constant_expr(*this)) == 1)
        return true;
    }
    else if (type_id == "floatbv")
    {
      if (ieee_floatt(to_constant_expr(*this)) == 1)
        return true;
    }
  }

  return false;
}

bool exprt::sum(const exprt &expr)
{
  if (!is_constant() || !expr.is_constant())
    return true;
  if (type() != expr.type())
    return true;

  const irep_idt &type_id = type().id();

  if (type_id == "unsignedbv" || type_id == "signedbv")
  {
    set(
      a_value,
      integer2binary(
        binary2integer(get_string(a_value), false) +
          binary2integer(expr.get_string(a_value), false),
        atoi(type().width().c_str())));
    return false;
  }
  if (type_id == "fixedbv")
  {
    fixedbvt f(to_constant_expr(*this));
    f += fixedbvt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }
  else if (type_id == "floatbv")
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
  if (!is_constant() || !expr.is_constant())
    return true;
  if (type() != expr.type())
    return true;

  const irep_idt &type_id = type().id();

  if (type_id == "unsignedbv" || type_id == "signedbv")
  {
    set(
      a_value,
      integer2binary(
        binary2integer(get_string(a_value), false) *
          binary2integer(expr.get_string(a_value), false),
        atoi(type().width().c_str())));
    return false;
  }
  if (type_id == "fixedbv")
  {
    fixedbvt f(to_constant_expr(*this));
    f *= fixedbvt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }
  else if (type_id == "floatbv")
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
  if (!is_constant() || !expr.is_constant())
    return true;

  if (type() != expr.type())
    return true;

  const irep_idt &type_id = type().id();

  if (type_id == "unsignedbv" || type_id == "signedbv")
  {
    set(
      a_value,
      integer2binary(
        binary2integer(get_string(a_value), false) -
          binary2integer(expr.get_string(a_value), false),
        atoi(type().width().c_str())));
    return false;
  }
  if (type_id == "fixedbv")
  {
    fixedbvt f(to_constant_expr(*this));
    f -= fixedbvt(to_constant_expr(expr));
    *this = f.to_expr();
    return false;
  }
  else if (type_id == "floatbv")
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

  if (l.is_not_nil())
    return l;

  forall_operands (it, (*this))
  {
    const locationt &l = it->find_location();
    if (l.is_not_nil())
      return l;
  }

  return static_cast<const locationt &>(get_nil_irep());
}

irep_idt exprt::trans = irep_idt("trans");
irep_idt exprt::symbol = irep_idt("symbol");
irep_idt exprt::plus = irep_idt("+");
irep_idt exprt::minus = irep_idt("-");
irep_idt exprt::mult = irep_idt("*");
irep_idt exprt::div = irep_idt("/");
irep_idt exprt::mod = irep_idt("mod");
irep_idt exprt::equality = irep_idt("=");
irep_idt exprt::notequal = irep_idt("notequal");
irep_idt exprt::index = irep_idt("index");
irep_idt exprt::arrayof = irep_idt("array_of");
irep_idt exprt::objdesc = irep_idt("object_descriptor");
irep_idt exprt::dynobj = irep_idt("dynamic_object");
irep_idt exprt::typecast = irep_idt("typecast");
irep_idt exprt::implies = irep_idt("=>");
irep_idt exprt::i_and = irep_idt("and");
irep_idt exprt::i_xor = irep_idt("xor");
irep_idt exprt::i_or = irep_idt("or");
irep_idt exprt::i_not = irep_idt("not");
irep_idt exprt::addrof = irep_idt("address_of");
irep_idt exprt::deref = irep_idt("dereference");
irep_idt exprt::i_if = irep_idt("if");
irep_idt exprt::with = irep_idt("with");
irep_idt exprt::member = irep_idt("member");
irep_idt exprt::member_ref = irep_idt("member_ref");
irep_idt exprt::ptr_mem = irep_idt("ptr_mem");
irep_idt exprt::isnan = irep_idt("isnan");
irep_idt exprt::ieee_floateq = irep_idt("ieee_float_equal");
irep_idt exprt::i_type = irep_idt("type");
irep_idt exprt::constant = irep_idt("constant");
irep_idt exprt::i_true = irep_idt("true");
irep_idt exprt::i_false = irep_idt("false");
irep_idt exprt::i_lt = irep_idt("<");
irep_idt exprt::i_gt = irep_idt(">");
irep_idt exprt::i_le = irep_idt("<=");
irep_idt exprt::i_ge = irep_idt(">=");
irep_idt exprt::i_cmp_three_way = irep_idt("<=>");
irep_idt exprt::i_bitand = irep_idt("bitand");
irep_idt exprt::i_bitor = irep_idt("bitor");
irep_idt exprt::i_bitxor = irep_idt("bitxor");
irep_idt exprt::i_bitnot = irep_idt("bitnot");
irep_idt exprt::i_ashr = irep_idt("ashr");
irep_idt exprt::i_lshr = irep_idt("lshr");
irep_idt exprt::i_shl = irep_idt("shl");
irep_idt exprt::abs = irep_idt("abs");
irep_idt exprt::argument = irep_idt("argument");

irep_idt exprt::a_value = irep_idt("value");

irep_idt exprt::o_operands = irep_idt("operands");
irep_idt exprt::o_location = irep_idt("#location");
