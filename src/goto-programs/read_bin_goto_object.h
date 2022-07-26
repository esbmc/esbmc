#ifndef READ_BIN_GOTO_OBJECT_H_
#define READ_BIN_GOTO_OBJECT_H_

#include <goto-programs/goto_functions.h>
#include <util/context.h>
#include <util/message.h>

/** Parses `in`. If failing to do so, a message is printed to `msg_hndlr`.
 *  @return true on error, false on success */
bool read_bin_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  goto_functionst &functions);

#endif /*READ_BIN_GOTO_OBJECT_H_*/
