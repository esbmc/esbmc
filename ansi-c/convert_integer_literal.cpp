/*******************************************************************\

Module: C++ Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <arith_tools.h>
#include <config.h>

#include "convert_integer_literal.h"

/*******************************************************************\

Function: convert_integer_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_integer_literal(
  const std::string &src,
  exprt &dest,
  unsigned base)
{
  bool is_unsigned=false;
  unsigned long_cnt=0;
  
  for(unsigned i=src.size(); i!=0; i--)
  {
    register char ch=src[i-1];
    if(ch=='u' || ch=='U')
      is_unsigned=true;
    else if(ch=='l' || ch=='L')
      long_cnt++;
    else
      break;
  }

  mp_integer value;
  
  if(base==10)
  {
    dest.value(src);
    value=string2integer(src, 10);
  }
  else if(base==8)
  {
    dest.hex_or_oct(true);
    value=string2integer(src, 8);
  }
  else
  {
    dest.hex_or_oct(true);
    std::string without_prefix(src, 2, std::string::npos);
    value=string2integer(without_prefix, 16);
  }
  
  typet type;
  
  if(is_unsigned)
    type=typet("unsignedbv");
  else
    type=typet("signedbv");

  if(long_cnt==0)
    type.set("width", config.ansi_c.int_width);
  else if(long_cnt==1)
    type.set("width", config.ansi_c.long_int_width);
  else
    type.set("width", config.ansi_c.long_long_int_width);
    
  dest=from_integer(value, type);

  dest.set("#cformat", src);
  
}
