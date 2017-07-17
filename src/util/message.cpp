/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/i2string.h>
#include <util/message.h>

void message_handlert::print(
  unsigned level,
  const std::string &message,
  const locationt &location)
{
  std::string dest;
  
  const irep_idt &file=location.get_file();
  const irep_idt &line=location.get_line();
  const irep_idt &column=location.get_column();
  const irep_idt &function=location.get_function();

  if(file!="")     { if(dest!="") dest+=" "; dest+="file "+id2string(file); }
  if(line!="")     { if(dest!="") dest+=" "; dest+="line "+id2string(line); }
  if(column!="")   { if(dest!="") dest+=" "; dest+="column "+id2string(column); }
  if(function!="") { if(dest!="") dest+=" "; dest+="function "+id2string(function); }

  if(dest!="") dest+=": ";
  dest+=message;

  print(level, dest);
}

void messaget::print(unsigned level, const std::string &message)
{
  if(message_handler!=nullptr && verbosity>=level)
    message_handler->print(level, message);
}
  
void messaget::print(
  unsigned level,
  const std::string &message,
  const locationt &location)
{
  if(message_handler!=nullptr && verbosity>=level)
    message_handler->print(level, message, location);
}
  
void messaget::set_message_handler(message_handlert *_message_handler)
{
  message_handler=_message_handler;
}
