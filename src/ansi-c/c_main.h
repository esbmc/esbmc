/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_MAIN_H
#define CPROVER_C_MAIN_H

#include <util/context.h>
#include <util/message.h>
#include <util/std_code.h>

bool c_main(
  contextt &context,
  const std::string &default_prefix,
  const std::string &standard_main,
  message_handlert &message_handler);

void static_lifetime_init(
  const contextt &context,
  codet &dest);

#endif
