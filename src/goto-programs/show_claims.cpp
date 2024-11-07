#include <goto-programs/show_claims.h>
#include <langapi/language_util.h>
#include <util/i2string.h>
#include <util/message/format.h>
#include <util/xml.h>
#include <util/xml_irep.h>

void show_claims(
  const namespacet &ns,
  const irep_idt &identifier,
  const goto_programt &goto_program,
  unsigned &count)
{
  for (const auto &instruction : goto_program.instructions)
  {
    if (instruction.is_assert())
    {
      count++;

      const irep_idt &comment = instruction.location.comment();
      const irep_idt description = (comment == "" ? "assertion" : comment);

      log_status(
        "Claim {}:\n  {}\n  {}\n  {}\n",
        count,
        instruction.location,
        description,
        from_expr(ns, identifier, instruction.guard));
    }

#ifdef ENABLE_JIMPLE_FRONTEND
    // In jimple asserts are modelled as throws (and try/catch is not really supported)
    if (instruction.is_throw())
    {
      count++;
      log_status(
        "Claim {}:\n  {}\n  {}\n  {}\n",
        count,
        instruction.location,
        "assertion",
        from_expr(ns, identifier, instruction.guard));
    }
#endif
  }
}

void show_claims(const namespacet &ns, const goto_programt &goto_program)
{
  unsigned count = 0;
  show_claims(ns, "", goto_program, count);
}

void show_claims(const namespacet &ns, const goto_functionst &goto_functions)
{
  unsigned count = 0;

  for (const auto &it : goto_functions.function_map)
    show_claims(ns, it.first, it.second.body, count);
}
