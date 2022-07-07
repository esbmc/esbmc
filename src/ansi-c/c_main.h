/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_C_MAIN_H
#define CPROVER_C_MAIN_H

#include <util/context.h>
#include <util/message.h>
#include <util/std_code.h>
#include <iostream>

bool c_main(contextt &context, const std::string &standard_main);

void static_lifetime_init(const contextt &context, codet &dest);

#endif
