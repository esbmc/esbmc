/*******************************************************************\

Module: Loop IDs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/loop_numbers.h>
#include <util/i2string.h>
#include <util/xml.h>
#include <util/xml_irep.h>
#include <util/message/format.h>

void show_loop_numbers(const goto_programt &goto_program, const messaget &msg)
{
  for(const auto &instruction : goto_program.instructions)
  {
    if(instruction.is_backwards_goto())
    {
      unsigned loop_id = instruction.loop_number;

      msg.debug(fmt::format("Loop {}:\n {}\n", loop_id, instruction.location));
    }
  }
}

void show_loop_numbers(
  const goto_functionst &goto_functions,
  const messaget &msg)
{
  for(const auto &it : goto_functions.function_map)
    show_loop_numbers(it.second.body, msg);
}
