/*******************************************************************\

Module: C++ Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/convert_integer_literal.h>
#include <cassert>
#include <util/arith_tools.h>
#include <util/config.h>

/*******************************************************************\

Function: convert_integer_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_integer_literal(
  const std::string &src,
  exprt &dest)
{
  bool is_unsigned=false;
  unsigned long_cnt=0;
  unsigned base=10;

  for(unsigned i=src.size(); i!=0; i--)
  {
    char ch=src[i-1];
    if(ch=='u' || ch=='U')
      is_unsigned=true;
    else if(ch=='l' || ch=='L')
      long_cnt++;
    else
      break;
  }

  mp_integer value;

  if(src.size()>=2 && src[0]=='0' && tolower(src[1])=='x')
  {
    // hex; strip "0x"
    base=16;
    dest.hex_or_oct(true);
    std::string without_prefix(src, 2, std::string::npos);
    value=string2integer(without_prefix, 16);
  }
  else if(src.size()>=2 && src[0]=='0')
  {
    // octal
    base=8;
    dest.hex_or_oct(true);
    value=string2integer(src, 8);
  }
  else
  {
    // The default is base 10.
    dest.value(src);
    value=string2integer(src, 10);
  }

  typet type;
  irep_idt cpp_type;

  if(is_unsigned)
    type=typet("unsignedbv");
  else
    type=typet("signedbv");

  mp_integer value_abs=value;

  if(value<0)
    value_abs.negate();

  bool is_hex_or_oct_or_bin=(base==8) || (base==16) || (base==2);

  #define FITS(width, signed) \
    ((signed?!is_unsigned:(is_unsigned || is_hex_or_oct_or_bin)) && \
    (power(2, signed?width-1:width)>value_abs))

  if(FITS(config.ansi_c.int_width, true) && long_cnt==0) // int
  {
    type.width(config.ansi_c.int_width);
    cpp_type="signed_int";
  }
  else if(FITS(config.ansi_c.int_width, false) && long_cnt==0) // unsigned int
  {
    type.width(config.ansi_c.int_width);
    cpp_type="unsigned_int";
  }
  else if(FITS(config.ansi_c.long_int_width, true) && long_cnt!=2) // long int
  {
    type.width(config.ansi_c.long_int_width);
    cpp_type="signed_long_int";
  }
  else if(FITS(config.ansi_c.long_int_width, false) && long_cnt!=2) // unsigned long int
  {
    type.width(config.ansi_c.long_int_width);
    cpp_type="unsigned_long_int";
  }
  else if(FITS(config.ansi_c.long_long_int_width, true)) // long long int
  {
    type.width(config.ansi_c.long_long_int_width);
    cpp_type="signed_long_long_int";
  }
  else if(FITS(config.ansi_c.long_long_int_width, false)) // unsigned long long int
  {
    type.width(config.ansi_c.long_long_int_width);
    cpp_type="unsigned_long_long_int";
  }
  else
  {
    // Way too large. Should consider issuing a warning.
    type.width(config.ansi_c.long_long_int_width);

    if(is_unsigned)
      cpp_type="unsigned_long_long_int";
    else
      cpp_type="signed_long_long_int";
  }

  type.set("#cpp_type", cpp_type);

  dest=from_integer(value, type);
  dest.cformat(src);
}
