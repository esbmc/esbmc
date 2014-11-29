/*******************************************************************\

Module: Show Claims

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <xml.h>
#include <i2string.h>
#include <xml_irep.h>

#include <langapi/language_util.h>

#include "show_claims.h"

void show_claims(
  const namespacet &ns,
  const irep_idt &identifier,
  ui_message_handlert::uit ui,
  const goto_programt &goto_program,
  unsigned &count)
{
  for(goto_programt::instructionst::const_iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();
      it++)
  {
    if(it->is_assert())
    {
      count++;

      const irep_idt &comment=it->location.comment();
      const irep_idt &property=it->location.property();
      const irep_idt description=
        (comment==""?"assertion":comment);

      if(ui==ui_message_handlert::XML_UI)
      {
        xmlt xml("claim");
        xml.new_element("number").data=i2string(count);
        
        xmlt &l=xml.new_element();
        convert(it->location, l);
        l.name="location";
        
        xml.new_element("description").data=
          xmlt::escape(id2string(description));
        
        xml.new_element("property").data=
          xmlt::escape(id2string(property));
        
        xml.new_element("expression").data=
          xmlt::escape(from_expr(ns, identifier, it->guard));
          
        std::cout << xml << std::endl;
      }
      else if(ui==ui_message_handlert::PLAIN)
      {
        std::cout << "Claim " << count << ":" << std::endl;

        std::cout << "  " << it->location << std::endl
                  << "  " << description
                  << std::endl;

        std::cout << "  " << from_expr(ns, identifier, it->guard)
                  << std::endl;
        std::cout << std::endl;
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
  unsigned count=0;
  show_claims(ns, "", ui, goto_program, count);
}

void show_claims(
  const namespacet &ns,
  ui_message_handlert::uit ui,
  const goto_functionst &goto_functions)
{
  unsigned count=0;

  for(goto_functionst::function_mapt::const_iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
    if(!it->second.is_inlined())
      show_claims(ns, it->first, ui, it->second.body, count);
}
