/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_CPROVER_LIBRARY_H
#define CPROVER_ANSI_C_CPROVER_LIBRARY_H

#include <util/context.h>
#include <util/message/message.h>

class languaget;

void add_cprover_library(
  contextt &context,
  const messaget &message_handler,
  const languaget *c_language = nullptr);

#endif
