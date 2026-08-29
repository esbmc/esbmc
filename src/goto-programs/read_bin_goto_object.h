#ifndef READ_BIN_GOTO_OBJECT_H_
#define READ_BIN_GOTO_OBJECT_H_

#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>
#include <util/message/message.h>
#include <vector>
#include <string>

/** Parses `in`. On failure an error is logged via log_error.
 *  @return true on error, false on success */
bool read_bin_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  goto_functionst &goto_functions);

/** As above, with a symbol whitelist and an out-context for what it filters
 *  out. `goto_functions` may be null to read the symbol table alone, which
 *  requires the stream's function-body section to be empty. */
bool read_bin_goto_object(
  std::istream &in,
  const std::string &filename,
  contextt &context,
  contextt &ignored,
  std::unordered_set<std::string> &function_set,
  goto_functionst *goto_functions);

#endif /*READ_BIN_GOTO_OBJECT_H_*/
