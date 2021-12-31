/*******************************************************************\

Module: Default message specialization. This should be used for
 common debug and CLI operations where the output is done through
 stdout/stderr

Author: Rafael Menezes, rafael.sa.menezes@outlook.com

Maintainers:
\*******************************************************************/

#include <memory>
#include <util/message/default_message.h>
#include <util/message/fmt_message_handler.h>

default_message::default_message()
{
  std::shared_ptr<message_handlert> handler =
    std::make_shared<fmt_message_handler>();
  this->add_message_handler(handler);
}