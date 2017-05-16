/*******************************************************************\

Module: ANSI-C Linking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_LINK_H
#define CPROVER_C_LINK_H

#include <util/context.h>
#include <util/message.h>

bool c_link(
  contextt &context,
  contextt &new_context,
  message_handlert &message_handler,
  const std::string &module);

#endif
