/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_CPROVER_LIBRARY_H
#define CPROVER_ANSI_C_CPROVER_LIBRARY_H

#include <util/context.h>
#include <util/message/message.h>

class languaget;

/* Returns the path the headers of the internal libc have been extracted to
 * or NULL if no library is configured (either via config.ansi_c.lib or during
 * build time). */
const std::string *internal_libc_header_dir();

void add_cprover_library(
  contextt &context,
  const messaget &message_handler,
  const languaget *c_language = nullptr);

#endif
