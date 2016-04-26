/*******************************************************************\

Module: C/C++ Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <arith_tools.h>

#include "c_types.h"
#include "unescape_string.h"
#include "convert_string_literal.h"
#include "string_constant.h"

/*******************************************************************\

Function: convert_string_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_string_literal(
  const std::string &src,
  std::string &dest)
{
  dest="";

  assert(src.size()>=2);
  assert(src[0]=='"' || src[0]=='L');
  assert(src[src.size()-1]=='"');

  if(src[0]=='L')
    unescape_string(std::string(src, 2, src.size()-3), dest);
  else
    unescape_string(std::string(src, 1, src.size()-2), dest);
}

/*******************************************************************\

Function: convert_string_literal

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_string_literal(const std::string &src, exprt &dest)
{
  std::string value;
  convert_string_literal(src, value);

  string_constantt result(value);

  dest.swap(result);
}
