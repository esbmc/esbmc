/*******************************************************************\

Module: Read Goto Programs

Author:

\*******************************************************************/

#include <goto-programs/read_bin_goto_object.h>
#include <goto-programs/read_goto_binary.h>

void read_goto_binary(
  std::istream &in,
  contextt &context,
  goto_functionst &dest,
  message_handlert &message_handler)
{
  read_bin_goto_object(in, "", context, dest, message_handler);
}
