/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cstdlib>
#include <util/arith_tools.h>
#include <util/std_expr.h>
#include <util/std_types.h>

unsigned bv_typet::get_width() const
{
  return atoi(get("width").c_str());
}

unsigned fixedbv_typet::get_integer_bits() const
{
  const std::string &integer_bits = get("integer_bits").as_string();
  assert(integer_bits != "");
  return atoi(integer_bits.c_str());
}

unsigned floatbv_typet::get_f() const
{
  const std::string &f = get_string("f");
  assert(f != "");
  return atoi(f.c_str());
}

unsigned
struct_union_typet::component_number(const irep_idt &component_name) const
{
  const componentst &c = components();

  unsigned number = 0;

  for(const auto &it : c)
  {
    if(it.get_name() == component_name)
      return number;

    number++;
  }

  assert(false);
  return 0;
}

const struct_union_typet::componentt &
struct_union_typet::get_component(const irep_idt &component_name) const
{
  const componentst &c = components();

  for(const auto &it : c)
  {
    //std::cout << "it->get_name(): " << it->get_name() << std::endl;
    //std::cout << "component_name: " << component_name << std::endl;
    if(it.get_name() == component_name)
      return it;
  }

  return static_cast<const componentt &>(get_nil_irep());
}

typet struct_union_typet::component_type(const irep_idt &component_name) const
{
  const exprt c = get_component(component_name);
  assert(c.is_not_nil());
  return c.type();
}

bool struct_typet::is_prefix_of(const struct_typet &other) const
{
  const componentst &ot_components = other.components();
  const componentst &tt_components = components();

  if(ot_components.size() < tt_components.size())
    return false;

  componentst::const_iterator ot_it = ot_components.begin();

  for(const auto &tt_component : tt_components)
  {
    if(
      ot_it->type() != tt_component.type() ||
      ot_it->name() != tt_component.name())
    {
      return false; // they just don't match
    }

    ot_it++;
  }

  return true; // ok, *this is a prefix of ot
}

bool is_reference(const typet &type)
{
  return type.id() == "pointer" && type.reference();
}

bool is_rvalue_reference(const typet &type)
{
  return type.id() == "pointer" && type.get_bool("#reference");
}

mp_integer signedbv_typet::smallest() const
{
  return -power(2, get_width() - 1);
}

mp_integer signedbv_typet::largest() const
{
  return power(2, get_width() - 1) - 1;
}

constant_exprt signedbv_typet::zero_expr() const
{
  return to_constant_expr(from_integer(0, *this));
}

constant_exprt signedbv_typet::smallest_expr() const
{
  return to_constant_expr(from_integer(smallest(), *this));
}

constant_exprt signedbv_typet::largest_expr() const
{
  return to_constant_expr(from_integer(largest(), *this));
}

mp_integer unsignedbv_typet::smallest() const
{
  return 0;
}

mp_integer unsignedbv_typet::largest() const
{
  return power(2, get_width()) - 1;
}

constant_exprt unsignedbv_typet::zero_expr() const
{
  return to_constant_expr(from_integer(0, *this));
}

constant_exprt unsignedbv_typet::smallest_expr() const
{
  return to_constant_expr(from_integer(smallest(), *this));
}

constant_exprt unsignedbv_typet::largest_expr() const
{
  return to_constant_expr(from_integer(largest(), *this));
}
