/*******************************************************************\

Module: Message Handler System. This system is responsible for all IO
  operations regarding the message system of ESBMC

Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#include <util/i2string.h>
#include <util/message/message_handler.h>

void message_handlert::print(
  VerbosityLevel level,
  const std::string &message,
  const locationt &location) const
{
  std::string dest;

  const irep_idt &file = location.get_file();
  const irep_idt &line = location.get_line();
  const irep_idt &column = location.get_column();
  const irep_idt &function = location.get_function();

  if(file != "")
  {
    if(dest != "")
      dest += " ";
    dest += "file " + id2string(file);
  }
  if(line != "")
  {
    if(dest != "")
      dest += " ";
    dest += "line " + id2string(line);
  }
  if(column != "")
  {
    if(dest != "")
      dest += " ";
    dest += "column " + id2string(column);
  }
  if(function != "")
  {
    if(dest != "")
      dest += " ";
    dest += "function " + id2string(function);
  }

  if(dest != "")
    dest += ": ";
  dest += message;

  print(level, dest);
}
