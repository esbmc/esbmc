/*******************************************************************\

Module: Default message specialization. This should be used for
 common debug and CLI operations where the output is done through
 stdout/stderr

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

Maintainers:
\*******************************************************************/

#ifndef ESBMC_DEFAULT_MESSAGE_H
#define ESBMC_DEFAULT_MESSAGE_H

#include <util/message/message.h>
class default_message : public messaget
{
public:
  default_message();

  static FILE *out;
  static FILE *err;
};

#endif //ESBMC_DEFAULT_MESSAGE_H
