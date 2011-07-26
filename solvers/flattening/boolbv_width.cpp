/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdlib.h>

#include <arith_tools.h>
#include <config.h>
#include <std_types.h>

#include "boolbv_width.h"

/*******************************************************************\

Function: boolbv_widtht::boolbv_widtht

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolbv_widtht::boolbv_widtht()
{
}

/*******************************************************************\

Function: boolbv_widtht::~boolbv_widtht

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

boolbv_widtht::~boolbv_widtht()
{
}

/*******************************************************************\

Function: boolbv_widtht::get_width

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolbv_widtht::get_width(const typet &type, unsigned &width) const
{
  if(type.id()=="bool")
  {
    width=1;
    return false;
  }
  else if(type.id()=="signedbv" ||
          type.id()=="unsignedbv" ||
          type.id()=="floatbv" ||
          type.id()=="fixedbv" ||
          type.id()=="bv")
  {
    width=atoi(type.width().c_str());
    assert(width!=0);
    return false;
  }
  else if(type.id()=="verilogbv")
  {
    width=atoi(type.width().c_str());
    width=width*2; // we encode with two bits
    assert(width!=0);
    return false;
  }
  else if(type.id()=="range")
  {
    mp_integer from=string2integer(type.from().as_string()),
                 to=string2integer(type.get_string("to"));

    mp_integer size=to-from+1;

    if(size<1)
      return true;

    width=integer2long(address_bits(size));
    assert(width!=0);

    return false;
  }
  else if(type.id()=="array")
  {
    const array_typet &array_type=to_array_type(type);
    unsigned sub_width;

    if(get_width(array_type.subtype(), sub_width))
      return true;

    mp_integer array_size;

    if(to_integer(array_type.size(), array_size))
    {
      // we can still use the theory of arrays for this
      width=0;
      return false;
    }

    width=integer2long(array_size*sub_width);

    return false;
  }
  else if(type.id()=="struct")
  {
    const irept &components=type.find("components");

    width=0;
    
    forall_irep(it, components.get_sub())
    {
      unsigned sub_width;
      const typet &sub_type=it->type();

      if(get_width(sub_type, sub_width))
        return true;

      width+=sub_width;
    }

    return false;
  }
  else if(type.id()=="code")
  {
    width=0;
    return false;
  }
  else if(type.id()=="union")
  {
    const irept &components=type.find("components");

    width=0;

    // get the biggest element, plus our index
    forall_irep(it, components.get_sub())
    {
      const typet &subtype=it->type();

      unsigned sub_width;
      if(get_width(subtype, sub_width))
        return true;

      width=std::max(width, sub_width);
    }

    width+=integer2long(address_bits(components.get_sub().size()));

    return false;
  }
  else if(type.id()=="enum")
  {
    width=0;

    // get number of necessary bits

    unsigned size=type.find("elements").get_sub().size();
    width=integer2long(address_bits(size));
    assert(width!=0);

    return false;
  }
  else if(type.id()=="c_enum" ||
          type.id()=="incomplete_c_enum")
  {
    width=atoi(type.width().c_str());
    assert(width!=0);
    return false;
  }
  else if(type.id()=="pointer" ||
          type.id()=="reference")
  {
    width=config.ansi_c.pointer_width+BV_ADDR_BITS;
    return false;
  }
  
  return true;
}

/*******************************************************************\

Function: boolbv_get_width

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool boolbv_get_width(const typet &type, unsigned &width)
{
  boolbv_widtht boolbv_width;
  return boolbv_width.get_width(type, width);
}

