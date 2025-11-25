#include <goto-programs/loop_numbers.h>
#include <util/i2string.h>
#include <util/message.h>
#include <util/message/format.h>
#include <util/xml.h>
#include <util/xml_irep.h>

void show_loop_numbers(
  const goto_programt &goto_program,
  const std::string &function_name)
{
  unsigned loop_index = 0;
  for (const auto &instruction : goto_program.instructions)
  {
    if (instruction.is_backwards_goto())
    {
      unsigned loop_id = instruction.loop_number;

      // Strip "c:@" prefix for cleaner user-facing output
      std::string display_name = function_name;
      if (display_name.substr(0, 3) == "c:@")
        display_name = display_name.substr(3);

      log_status(
        "goto-loop Loop {} ({}:{}):\n {}\n",
        loop_id,
        display_name,
        loop_index,
        instruction.location);

      loop_index++;
    }
  }
}

void show_loop_numbers(const goto_functionst &goto_functions)
{
  for (const auto &it : goto_functions.function_map)
    show_loop_numbers(it.second.body, it.first.as_string());
}
