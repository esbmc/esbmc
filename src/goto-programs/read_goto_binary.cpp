/*******************************************************************\

Module: Read Goto Programs

Author:

\*******************************************************************/

#include <goto-programs/read_bin_goto_object.h>
#include <goto-programs/read_goto_binary.h>
#include <fstream>

bool read_goto_binary(
  const std::string &path,
  contextt &context,
  goto_functionst &dest,
  const messaget &msg)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  return read_bin_goto_object(in, path, context, dest, msg);
}
