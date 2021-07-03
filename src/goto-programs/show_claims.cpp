/*******************************************************************\

Module: Show Claims

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/show_claims.h>
#include <langapi/language_util.h>
#include <util/i2string.h>
#include <util/xml.h>
#include <util/xml_irep.h>
#include <util/message/format.h>

void show_claims(
  const namespacet &ns,
  const irep_idt &identifier,
  const goto_programt &goto_program,
  unsigned &count,
  const messaget &msg)
{
  for(const auto &instruction : goto_program.instructions)
  {
    if(instruction.is_assert())
    {
      count++;

      const irep_idt &comment = instruction.location.comment();
      const irep_idt description = (comment == "" ? "assertion" : comment);

      msg.status(fmt::format(
        "Claim {}:\n  {}\n  {}\n  {}\n",
        count,
        instruction.location,
        description,
        from_expr(ns, identifier, instruction.guard, msg)));
    }
  }
}

void show_claims(
  const namespacet &ns,
  const goto_programt &goto_program,
  const messaget &msg)
{
  unsigned count = 0;
  show_claims(ns, "", goto_program, count, msg);
}

void show_claims(
  const namespacet &ns,
  const goto_functionst &goto_functions,
  const messaget &msg)
{
  unsigned count = 0;

  for(const auto &it : goto_functions.function_map)
    show_claims(ns, it.first, it.second.body, count, msg);
}
