/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/i2string.h>
#include <util/parser.h>

#ifdef _WIN32
int isatty(int f)
{
  return 0;
}
#endif

exprt &_newstack(parsert &parser, unsigned &x)
{
  x = parser.stack.size();

  if(x >= parser.stack.capacity())
    parser.stack.reserve(x * 2);

  parser.stack.push_back(static_cast<const exprt &>(get_nil_irep()));
  return parser.stack.back();
}

void parsert::parse_error(const std::string &message, const std::string &before)
{
  locationt location;
  location.set_file(filename);
  location.set_line(i2string(line_no));
  std::string tmp = message;
  if(before != "")
    tmp += " before `" + before + "'";
  print(1, tmp, location);
}
