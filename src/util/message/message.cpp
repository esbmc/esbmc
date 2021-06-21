/*******************************************************************\

Module: Message System. This system is used to send messages through
    ESBMC execution.
Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#include <util/message/message.h>

template <typename... TS>
std::string
messaget::format_to_string(const std::string &format, const TS &...ts)
{
  return "NOT IMPLEMENTED";
}
