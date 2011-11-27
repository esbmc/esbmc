/*******************************************************************\

Module: String Abstraction

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_GOTO_PROGRAMS_STRING_ABSTRACTION_H
#define CPROVER_GOTO_PROGRAMS_STRING_ABSTRACTION_H

#include <context.h>
#include <message_stream.h>

#include "goto_functions.h"

// keep track of length of strings

void string_abstraction(
  contextt &context,
  message_handlert &message_handler,
  goto_functionst &dest);

#endif
