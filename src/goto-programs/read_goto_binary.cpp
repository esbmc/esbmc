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
  const messaget &message_handler)
{
  bool fail [[gnu::unused]] =
    read_bin_goto_object(in, "", context, dest, message_handler);
  assert(!fail);
}
