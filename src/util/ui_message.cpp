/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <fstream>
#include <util/i2string.h>
#include <util/ui_message.h>
#include <util/xml.h>
#include <util/xml_irep.h>

const char *ui_message_handlert::level_string(unsigned level)
{
  if(level == 1)
    return "ERROR";
  if(level == 2)
    return "WARNING";
  else
    return "STATUS-MESSAGE";
}

void ui_message_handlert::print(unsigned level, const std::string &message)
{
  
    if(level == 1)
      std::cerr << message << std::endl;
    else
      std::cout << message << std::endl;
}

void ui_message_handlert::print(
  unsigned level,
  const std::string &message,
  const locationt &location)
{  
    message_handlert::print(level, message, location);
}
