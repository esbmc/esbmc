/*******************************************************************\

Module: 

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdlib.h>

#include "std_types.h"

/*******************************************************************\

Function: bv_typet::get_width

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned bv_typet::get_width() const
{
  return atoi(get("width").c_str());

}

/*******************************************************************\

Function: fixedbv_typet::get_integer_bits

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned fixedbv_typet::get_integer_bits() const
{
  const std::string &integer_bits=get("integer_bits").as_string();
  assert(integer_bits!="");
  return atoi(integer_bits.c_str());
}

/*******************************************************************\

Function: floatbv_typet::get_f

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned floatbv_typet::get_f() const
{
  const std::string &f=get_string("f");
  assert(f!="");
  return atoi(f.c_str());
}

/*******************************************************************\

Function: struct_union_typet::component_number

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned struct_union_typet::component_number(
  const irep_idt &component_name) const
{
  const componentst &c=components();

  unsigned number=0;

  for(componentst::const_iterator
      it=c.begin();
      it!=c.end();
      it++)
  {
    if(it->get_name()==component_name)
      return number;

    number++;
  }

  assert(false);
  return 0;
}

/*******************************************************************\

Function: struct_union_typet::get_component

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const struct_union_typet::componentt &struct_union_typet::get_component(
  const irep_idt &component_name) const
{
  const componentst &c=components();

  for(componentst::const_iterator
      it=c.begin();
      it!=c.end();
      it++)
  {
    if(it->get_name()==component_name)
      return *it;
  }

  return static_cast<const componentt &>(get_nil_irep());
}

/*******************************************************************\

Function: struct_union_typet::component_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

typet struct_union_typet::component_type(
  const irep_idt &component_name) const
{
  const exprt c=get_component(component_name);
  assert(c.is_not_nil());
  return c.type();
}

/*******************************************************************\

Function: struct_typet::is_prefix_of

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool struct_typet::is_prefix_of(const struct_typet &other) const
{
  const componentst &ot_components=other.components();
  const componentst &tt_components=components();

  if(ot_components.size()<
     tt_components.size())
    return false; 

  componentst::const_iterator
    ot_it=ot_components.begin();

  for(componentst::const_iterator tt_it=
      tt_components.begin();
      tt_it!=tt_components.end();
      tt_it++)
  {
    if(ot_it->type()!=tt_it->type() ||
       ot_it->name()!=tt_it->name())
    {
      return false; // they just don't match
    }

    ot_it++;
  }

  return true; // ok, *this is a prefix of ot
}

/*******************************************************************\

Function: is_reference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool is_reference(const typet &type)
{
  return type.id()=="pointer" && type.reference();
}

/*******************************************************************\

Function: is_rvalue_reference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool is_rvalue_reference(const typet &type)
{
  return type.id()=="pointer" &&
         type.get_bool("#reference");
}
