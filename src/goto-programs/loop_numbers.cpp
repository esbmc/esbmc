/*******************************************************************\

Module: Loop IDs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/loop_numbers.h>
#include <util/i2string.h>
#include <util/xml.h>
#include <util/xml_irep.h>

void show_loop_numbers(
  ui_message_handlert::uit ui,
  const goto_programt &goto_program)
{
  for(const auto &instruction : goto_program.instructions)
  {
    if(instruction.is_backwards_goto())
    {
      unsigned loop_id = instruction.loop_number;

      if(ui == ui_message_handlert::XML_UI)
      {
        xmlt xml("loop");
        xml.new_element("loop-id").data = i2string(loop_id);

        xmlt &l = xml.new_element();
        convert(instruction.location, l);
        l.name = "location";

        std::cout << xml << std::endl;
      }
      else if(ui == ui_message_handlert::PLAIN)
      {
        std::cout << "Loop " << loop_id << ":" << std::endl;

        std::cout << "  " << instruction.location << std::endl;
        std::cout << std::endl;
      }
      else
        assert(false);
    }
  }
}

void show_loop_numbers(
  ui_message_handlert::uit ui,
  const goto_functionst &goto_functions)
{
  for(const auto &it : goto_functions.function_map)
    show_loop_numbers(ui, it.second.body);
}
