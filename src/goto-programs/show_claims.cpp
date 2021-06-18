/*******************************************************************\

Module: Show Claims

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/show_claims.h>
#include <langapi/language_util.h>
#include <util/i2string.h>
#include <util/xml.h>
#include <util/xml_irep.h>

void show_claims(
  const namespacet &ns,
  const irep_idt &identifier,
  ui_message_handlert::uit ui,
  const goto_programt &goto_program,
  unsigned &count)
{
  for(const auto &instruction : goto_program.instructions)
  {
    if(instruction.is_assert())
    {
      count++;

      const irep_idt &comment = instruction.location.comment();
      const irep_idt &property = instruction.location.property();
      const irep_idt description = (comment == "" ? "assertion" : comment);

      if(ui == ui_message_handlert::XML_UI)
      {
        xmlt xml("claim");
        xml.new_element("number").data = i2string(count);

        xmlt &l = xml.new_element();
        convert(instruction.location, l);
        l.name = "location";

        xml.new_element("description").data =
          xmlt::escape(id2string(description));

        xml.new_element("property").data = xmlt::escape(id2string(property));

        xml.new_element("expression").data =
          xmlt::escape(from_expr(ns, identifier, instruction.guard));

        PRINT(xml << "\n");
      }
      else if(ui == ui_message_handlert::PLAIN)
      {
        PRINT(
          "Claim " << count << ":"
                   << "\n");

        PRINT(
          "  " << instruction.location << "\n"
               << "  " << description << "\n");

        PRINT("  " << from_expr(ns, identifier, instruction.guard) << "\n");
        PRINT("\n");
      }
      else
        assert(false);
    }
  }
}

void show_claims(
  const namespacet &ns,
  ui_message_handlert::uit ui,
  const goto_programt &goto_program)
{
  unsigned count = 0;
  show_claims(ns, "", ui, goto_program, count);
}

void show_claims(
  const namespacet &ns,
  ui_message_handlert::uit ui,
  const goto_functionst &goto_functions)
{
  unsigned count = 0;

  for(const auto &it : goto_functions.function_map)
    show_claims(ns, it.first, ui, it.second.body, count);
}
