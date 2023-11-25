/*******************************************************************\

Module: C/C++ Language Conversion

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/convert_string_literal.h>
#include <ansi-c/unescape_string.h>
#include <cassert>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/string_constant.h>

void convert_string_literal(const std::string &src, std::string &dest)
{
  dest = "";

  assert(src.size() >= 2);
  assert(src[0] == '"' || src[0] == 'L');
  assert(src[src.size() - 1] == '"');

  if (src[0] == 'L')
    unescape_string(std::string(src, 2, src.size() - 3), dest);
  else
    unescape_string(std::string(src, 1, src.size() - 2), dest);
}

void convert_string_literal(const std::string &src, exprt &dest)
{
  std::string value;
  convert_string_literal(src, value);

  string_constantt result(value);

  dest.swap(result);
}
