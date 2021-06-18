/*******************************************************************\

Module: Show Value Sets

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <pointer-analysis/show_value_sets.h>

void show_value_sets(
  ui_message_handlert::uit ui,
  const goto_functionst &goto_functions,
  const value_set_analysist &value_set_analysis)
{
  switch(ui)
  {
  case ui_message_handlert::XML_UI:
  {
    xmlt xml;
    convert(goto_functions, value_set_analysis, xml);
    PRINT(xml << "\n");
  }
  break;

  case ui_message_handlert::PLAIN:
  {
    std::stringstream out;
    value_set_analysis.output(goto_functions, out);
    PRINT(out.str());
  }
  break;

  default:;
  }
}

void show_value_sets(
  ui_message_handlert::uit ui,
  const goto_programt &goto_program,
  const value_set_analysist &value_set_analysis)
{
  switch(ui)
  {
  case ui_message_handlert::XML_UI:
  {
    xmlt xml;
    convert(goto_program, value_set_analysis, xml);
    PRINT(xml << "\n");
  }
  break;

  case ui_message_handlert::PLAIN:
  {
    std::stringstream out;
    value_set_analysis.output(goto_program, out);
    PRINT(out.str());
  }
  break;

  default:;
  }
}
