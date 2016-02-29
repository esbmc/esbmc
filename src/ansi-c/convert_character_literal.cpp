/*******************************************************************\

Module: C Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <arith_tools.h>
#include <i2string.h>
#include <c_types.h>

#include <ansi-c/unescape_string.h>

#include "convert_character_literal.h"

/*******************************************************************\

Function: convert_character_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_character_literal(
  const std::string &src,
  std::string &dest)
{
  dest="";

  assert(src.size()>=2);
  assert(src[0]=='\'' || src[0]=='L');
  assert(src[src.size()-1]=='\'');

  if(src[0]=='L')
    unescape_string(std::string(src, 2, src.size()-3), dest);
  else
    unescape_string(std::string(src, 1, src.size()-2), dest);
}

/*******************************************************************\

Function: convert_character_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_character_literal(
  const std::string &src,
  exprt &dest)
{
  std::string value;
  convert_character_literal(src, value);

  //std::cout << "src.c_str(): " << src.c_str() << std::endl;
  //std::cout << "value.size(): " << value.size() << std::endl;

  typet type;
  irep_idt cpp_type;

  type=typet("unsignedbv");

  if(value.size()==0)
    throw "empty character literal";
  else if(value.size()==1)
  {
    type = char_type();
    type.set("#cpp_type", "char");
    dest=from_integer(value[0], type);
  }
  else if(value.size()>=2 && value.size()<=4)
  {
    mp_integer x=0;

    for(unsigned i=0; i<value.size(); i++)
    {
      mp_integer z=(unsigned char)(value[i]);
      z=z<<((value.size()-i-1)*8);
      x+=z;
    }

    dest=from_integer(x, int_type());
  }
  else
  {
    throw "literals with "+i2string((unsigned long)value.size())+
          " characters are not supported";
  }
}
