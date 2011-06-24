/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "parser.h"
#include "i2string.h"

#ifdef _WIN32
int isatty(int f)
{
  return 0;
}
#endif

/*******************************************************************\

Function:

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void parsert::parse_error(
  const std::string &message,
  const std::string &before)
{
  locationt location;
  location.set_file(filename);
  location.set_line(i2string(line_no));
  std::string tmp=message;
  if(before!="") tmp+=" before `"+before+"'";
  print(1, tmp, -1, location);
}

