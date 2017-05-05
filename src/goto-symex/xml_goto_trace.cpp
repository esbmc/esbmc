/*******************************************************************\

Module: Traces of GOTO Programs

Author: Daniel Kroening

  Date: November 2005

\*******************************************************************/

#include <ansi-c/printf_formatter.h>
#include <cassert>
#include <goto-symex/xml_goto_trace.h>
#include <langapi/language_util.h>
#include <util/i2string.h>
#include <util/xml_irep.h>

void convert(
  const namespacet &ns,
  const goto_tracet &goto_trace,
  xmlt &xml)
{
  xml=xmlt("goto_trace");
  
  xml.new_element("mode").data=goto_trace.mode;

  locationt previous_location;

  for(goto_tracet::stepst::const_iterator
      it=goto_trace.steps.begin();
      it!=goto_trace.steps.end();
      it++)
  {
    const locationt &location=it->pc->location;
    
    xmlt xml_location;
    if(location.is_not_nil() && location.get_file()!="")
    {
      convert(location, xml_location);
      xml_location.name="location";
    }
    
    switch(it->type)
    {
    case goto_trace_stept::ASSERT:
      if(!it->guard)
      {
        xmlt &xml_failure=xml.new_element("failure");
        xml_failure.new_element("reason").data=xmlt::escape(id2string(it->comment));
        
        xml_failure.new_element("thread").data=i2string(it->thread_nr);

        if(xml_location.name!="")
          xml_failure.new_element().swap(xml_location);
      }
      break;
      
    case goto_trace_stept::ASSIGNMENT:
      {
        irep_idt identifier;

        if (!is_nil_expr(it->original_lhs))
          identifier = to_symbol2t(it->original_lhs).get_symbol_name();
        else
          identifier = to_symbol2t(it->lhs).get_symbol_name();
          
        xmlt &xml_assignment=xml.new_element("assignment");

        if(xml_location.name!="")
          xml_assignment.new_element().swap(xml_location);

        std::string value_string, type_string;
        
        if (!is_nil_expr(it->value)) {
          value_string = from_expr(ns, identifier,
                                   migrate_expr_back(it->value));
          type_string=from_type(ns, identifier,
                                migrate_type_back(it->value->type));
        }

        const symbolt *symbol;
        irep_idt base_name, display_name;

        if(!ns.lookup(identifier, symbol))
        {
          base_name=symbol->base_name;
          display_name=symbol->display_name();
          if(type_string=="")
            type_string=from_type(ns, identifier, symbol->type);

          xml_assignment.new_element("mode").data=xmlt::escape(id2string(symbol->mode));
        }

        xml_assignment.new_element("thread").data=i2string(it->thread_nr);
        xml_assignment.new_element("identifier").data=xmlt::escape(id2string(identifier));
        xml_assignment.new_element("base_name").data=xmlt::escape(id2string(base_name));
        xml_assignment.new_element("display_name").data=xmlt::escape(id2string(display_name));
        xml_assignment.new_element("value").data=xmlt::escape(id2string(value_string));
        xml_assignment.new_element("type").data=xmlt::escape(id2string(type_string));
        xml_assignment.new_element("step_nr").data=i2string(it->step_nr);
      }
      break;
      
    case goto_trace_stept::OUTPUT:
      {
        printf_formattert printf_formatter;

        std::list<exprt> vec;

        for (std::list<expr2tc>::const_iterator it2 = it->output_args.begin();
             it2 != it->output_args.end(); it2++) {
          vec.push_back(migrate_expr_back(*it2));
        }

        printf_formatter(it->format_string, vec);
        std::string text=printf_formatter.as_string();
        xmlt &xml_output=xml.new_element("output");
        xml_output.new_element("step_nr").data=i2string(it->step_nr);
        xml_output.new_element("thread").data=i2string(it->thread_nr);
        xml_output.new_element("text").data=xmlt::escape(text);
        xml_output.new_element().swap(xml_location);
      }
      break;
      
    default:
      if(location!=previous_location)
      {
        // just the location
        if(xml_location.name!="")
        {
          xmlt &xml_location_only=xml.new_element("location-only");
          xml_location_only.new_element("step_nr").data=i2string(it->step_nr);
          xml_location_only.new_element("thread").data=i2string(it->thread_nr);
          xml_location_only.new_element().swap(xml_location);
        }
      }
    }

    if(location.is_not_nil() && location.get_file()!="")
      previous_location=location;
  }
}
