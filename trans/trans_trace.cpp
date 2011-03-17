/*******************************************************************\

Module: Extracting Counterexamples

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>
#include <string>
#include <xml.h>
#include <xml_irep.h>
#include <i2string.h>
#include <expr_util.h>
#include <prefix.h>

#include <langapi/language_util.h>

#include "instantiate.h"
#include "trans_trace.h"

/*******************************************************************\

Function: compute_trans_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void compute_trans_trace(
  const decision_proceduret &decision_procedure,
  unsigned no_timeframes,
  const namespacet &ns,
  const irep_idt &module,
  trans_tracet &dest)
{
  // look up the module symbol
  {
    const symbolt &symbol=ns.lookup(module);
    dest.mode=id2string(symbol.mode);
  }

  dest.states.resize(no_timeframes);

  for(unsigned t=0; t<no_timeframes; t++)
  {
    const contextt &context=ns.get_context();

    assert(t<dest.states.size());
    trans_tracet::statet &state=dest.states[t];
    
    forall_symbol_module_map(it, context.symbol_module_map, module)
    {
      const symbolt &symbol=ns.lookup(it->second);

      if(!symbol.is_type &&
         !symbol.theorem &&
         symbol.type.id()!="module" &&
         symbol.type.id()!="module_instance")
      {
        exprt indexed_symbol_expr("symbol", symbol.type);

        indexed_symbol_expr.set("identifier",
          timeframe_identifier(t, symbol.name));

        exprt value_expr=decision_procedure.get(indexed_symbol_expr);

        trans_tracet::statet::assignmentt assignment;
        assignment.rhs.swap(value_expr);
        assignment.lhs=symbol_expr(symbol);
      
        state.assignments.push_back(assignment);
      }
    }
  }
}

/*******************************************************************\

Function: compute_trans_trace_properties

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void compute_trans_trace_properties(
  const std::list<bvt> &prop_bv,
  const propt &solver,
  unsigned no_timeframes,
  trans_tracet &dest)  
{
  // check the properties that got violated

  for(std::list<bvt>::const_iterator
      p_it=prop_bv.begin();
      p_it!=prop_bv.end();
      p_it++)
  {
    dest.properties.push_back(trans_tracet::propertyt());

    const bvt &bv=*p_it;
    assert(bv.size()==no_timeframes);
    
    bool saw_unknown=false,
         saw_failure=false;
  
    for(unsigned t=0; t<no_timeframes; t++)
    {
      tvt result=solver.l_get(bv[t]);

      if(result.is_unknown())
      {
        saw_unknown=true;
      }
      else if(result.is_false())
      {
        dest.properties.back().failing_timeframe=t;
        saw_failure=true;
        break; // next property
      }
    }

    if(saw_failure)
      dest.properties.back().status=tvt(false);
    else if(saw_unknown)
      dest.properties.back().status=tvt(tvt::TV_UNKNOWN);
    else
      dest.properties.back().status=tvt(true);
  }
}

/*******************************************************************\

Function: compute_trans_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void compute_trans_trace(
  const std::list<exprt> &properties,
  const std::list<bvt> &prop_bv,
  const class prop_convt &solver,
  unsigned no_timeframes,
  const namespacet &ns,
  const irep_idt &module,
  trans_tracet &dest)  
{
  compute_trans_trace(
    solver,
    no_timeframes,
    ns,
    module,
    dest);
    
  // check the properties that got violated
  
  assert(properties.size()==prop_bv.size());

  compute_trans_trace_properties(
    prop_bv,
    solver.prop,
    no_timeframes,
    dest);
}

/*******************************************************************\

Function: compute_trans_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void compute_trans_trace(
  const std::list<bvt> &prop_bv,
  const bmc_mapt &bmc_map,
  const class propt &solver,
  const namespacet &ns,
  trans_tracet &dest)
{
  dest.states.reserve(bmc_map.get_no_timeframes());
  
  for(unsigned t=0; t<bmc_map.get_no_timeframes(); t++)
  {
    dest.states.push_back(trans_tracet::statet());
    trans_tracet::statet &state=dest.states.back();
  
    for(var_mapt::mapt::const_iterator
        it=bmc_map.var_map.map.begin();
        it!=bmc_map.var_map.map.end();
        it++)
    {
      const var_mapt::vart &var=it->second;
      
      if(var.vartype!=var_mapt::vart::VAR_LATCH &&
         var.vartype!=var_mapt::vart::VAR_INPUT &&
         var.vartype!=var_mapt::vart::VAR_WIRE)
        continue;
        
      const symbolt &symbol=ns.lookup(it->first);

      std::string value;

      for(unsigned i=0; i<var.bits.size(); i++)
      {
        literalt l=bmc_map.get(t, var.bits[i]);

        char ch;

        switch(solver.l_get(l).get_value())
        {
         case tvt::TV_TRUE: ch='1'; break;
         case tvt::TV_FALSE: ch='0'; break;
         default: ch='?'; break;
        }

        value=ch+value;
      }
      
      exprt value_expr;
      value_expr.make_nil();

      if(var.type.id()=="range" ||
         var.type.id()=="unsignedbv" ||
         var.type.id()=="signedbv")
      {
        value_expr=exprt("constant", var.type);

        if(var.type.id()=="range")
        {
          mp_integer i=binary2integer(value, false);
          mp_integer from=string2integer(var.type.get_string("from"));
          value_expr.set("value", integer2string(i+from));
        }
        else
          value_expr.set("value", value);
      }
      else if(var.type.id()=="bool")
      {
        if(value=="0")
          value_expr.make_false();
        else if(value=="1")
          value_expr.make_true();
      }
      
      if(value_expr.is_not_nil())
      {
        state.assignments.push_back(trans_tracet::statet::assignmentt());

        trans_tracet::statet::assignmentt &assignment=
          state.assignments.back();

        assignment.lhs=symbol_expr(symbol);
        assignment.rhs=value_expr;
        assignment.location.make_nil();
      }
    }
  }

  // check the properties that got violated
  
  compute_trans_trace_properties(
    prop_bv,
    solver,
    bmc_map.get_no_timeframes(),
    dest);
}         
          
/*******************************************************************\

Function: show_trans_state

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_trans_state(
  unsigned timeframe,
  const trans_tracet::statet &state,
  const namespacet &ns)
{
  std::cout << "Transition system state " << timeframe << "\n";
  std::cout << "----------------------------------------------------\n";

  for(trans_tracet::statet::assignmentst::const_iterator
      it=state.assignments.begin();
      it!=state.assignments.end();
      it++)
  {
    assert(it->lhs.id()=="symbol");

    const symbolt &symbol=ns.lookup(it->lhs.get("identifier"));

    std::cout << "  " << symbol.display_name() << " = ";

    const exprt &rhs=it->rhs;

    if(rhs.is_nil())
      std::cout << "?";
    else
      std::cout << from_expr(ns, symbol.name, rhs);
    
    if(rhs.type().id()=="unsignedbv" ||
       rhs.type().id()=="signedbv" ||
       rhs.type().id()=="bv")
    {
      unsigned width=atoi(rhs.type().get("width").c_str());
      
      if(width>=2 && width<=32 &&
         rhs.id()=="constant")
        std::cout << " (" << rhs.get("value") << ")";
    }
    
    std::cout << std::endl;
  }

  std::cout << std::endl;
}

/*******************************************************************\

Function: convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert(
  const trans_tracet &trace,
  unsigned last_time_frame,
  const namespacet &ns,
  xmlt &dest)
{
  dest=xmlt("trans_trace");
  
  dest.new_element("mode").data=trace.mode;

  for(unsigned t=0; t<=last_time_frame; t++)
  {
    assert(t<trace.states.size());
  
    xmlt &xml_state=dest.new_element("state");
    const trans_tracet::statet &state=trace.states[t];

    xml_state.new_element("timeframe").data=i2string(t);
    
    for(trans_tracet::statet::assignmentst::const_iterator
        it=state.assignments.begin();
        it!=state.assignments.end();
        it++)
    {
      xmlt &xml_assignment=xml_state.new_element("assignment");

      assert(it->lhs.id()=="symbol");
      const symbolt &symbol=ns.lookup(it->lhs.get("identifier"));

      std::string value_string=from_expr(ns, symbol.name, it->rhs);
      std::string type_string=from_type(ns, symbol.name, symbol.type);

      if(it->rhs.is_nil())
        value_string="?";

      xml_assignment.new_element("identifier").data=xmlt::escape(id2string(symbol.name));
      xml_assignment.new_element("base_name").data=xmlt::escape(id2string(symbol.base_name));
      xml_assignment.new_element("display_name").data=xmlt::escape(id2string(symbol.display_name()));
      xml_assignment.new_element("value").data=xmlt::escape(value_string);
      xml_assignment.new_element("type").data=xmlt::escape(type_string);
      xml_assignment.new_element("mode").data=xmlt::escape(id2string(symbol.mode));
      
      if(it->location.is_not_nil())
      {
        xmlt &xml_location=xml_assignment.new_element();

        convert(it->location, xml_location);
        xml_location.name="location";
      }
    }
  }

  {
    unsigned p=1;

    for(trans_tracet::propertiest::const_iterator
        p_it=trace.properties.begin();
        p_it!=trace.properties.end();
        p_it++, p++)
    {
      xmlt &xml_claim_status=dest.new_element("claim-status");
      
      xml_claim_status.new_element("claim").data=i2string(p);
      
      if(p_it->status.is_false())
      {
        xml_claim_status.new_element("time_frame").data=
          i2string(p_it->failing_timeframe);

        xml_claim_status.new_element("status").data="false";
      }
      else if(p_it->status.is_true())
        xml_claim_status.new_element("status").data="true";
      else
        xml_claim_status.new_element("status").data="unknown";
    }
  }
}

/*******************************************************************\

Function: show_trans_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void show_trans_trace(
  const trans_tracet &trace,
  messaget &message,
  const namespacet &ns,
  ui_message_handlert::uit ui)
{
  unsigned l=trace.get_failing_timeframe();

  switch(ui)
  {
  case ui_message_handlert::PLAIN:
    for(unsigned t=0; t<=l; t++)
      show_trans_state(t, trace.states[t], ns);

    {
      unsigned p=1;

      for(trans_tracet::propertiest::const_iterator
          p_it=trace.properties.begin();
          p_it!=trace.properties.end();
          p_it++, p++)
        if(p_it->status.is_false())
        {
          std::cout << "Property " << p << " violated in "
                       "time frame " << p_it->failing_timeframe
                    << std::endl;
        }
    }
    break;
    
  case ui_message_handlert::XML_UI:
    {
      xmlt xml;
      
      convert(trace, l, ns, xml);
      
      xml.output(std::cout);
    }
    break;
    
  default:
    assert(false);
  }
}

/*******************************************************************\

Function: vcd_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string vcd_identifier(const std::string &id)
{
  std::string result=id;

  if((has_prefix(result, "verilog::")) || (has_prefix(result, "Verilog::")))
    result.erase(0, 9);
  else if(has_prefix(result, "smv::"))
    result.erase(0, 5);
    
  return result;
}

/*******************************************************************\

Function: show_trans_trace_vcd

  Inputs:

 Outputs:

 Purpose: dumps the counterexample state in vcd format to be
          viewed in modelsim or any other simulator

\*******************************************************************/

void show_trans_state_vcd(
  unsigned timeframe,
  const trans_tracet::statet &state,
  const namespacet &ns,
  std::ostream &out)
{
  out << "#" << timeframe << std::endl;

  for(trans_tracet::statet::assignmentst::const_iterator
      it=state.assignments.begin();
      it!=state.assignments.end();
      it++)
  {
    assert(it->lhs.id()=="symbol");

    const symbolt &symbol=ns.lookup(it->lhs.get("identifier"));

    std::string value;

    if(it->rhs.type().id()=="unsignedbv" ||
       it->rhs.type().id()=="signedbv" ||
       it->rhs.type().id()=="verilogbv")
    {
      if(it->rhs.is_not_nil())
	value="b"+id2string(it->rhs.get("value"));
      else
	value="X";
    }
    else
    {
      if(it->rhs.is_not_nil())
        value=from_expr(ns, symbol.name, it->rhs);
      else
        value="0";
    }   

    std::string display_name=id2string(symbol.display_name());
    out << value << " " << vcd_identifier(display_name) << std::endl;
  }
}

/*******************************************************************\

Function: show_trans_trace_vcd

  Inputs:

 Outputs:

 Purpose: dumps the counterexample trace in vcd format to be
          viewed in modelsim or any other simulator

\*******************************************************************/

void show_trans_trace_vcd(
  const trans_tracet &trace,
  messaget &message,
  const namespacet &ns,
  ui_message_handlert::uit ui,
  std::ostream &out)
{
  out << "$timescale \n 1ns\n$end\n";
  
  if(trace.states.empty()) return;

  const trans_tracet::statet &state=trace.states[0];

  assert(!state.assignments.empty());

  const symbolt &symbol1=ns.lookup(
    state.assignments.front().lhs.get("identifier"));

  std::string module_name=id2string(symbol1.module);
  out << "$scope module " << vcd_identifier(module_name) << " $end\n";

  std::list<irep_idt> last_hierarchy;

  for(trans_tracet::statet::assignmentst::const_iterator
      it=state.assignments.begin();
      it!=state.assignments.end();
      it++)
  {
    assert(it->lhs.id()=="symbol");

    const symbolt &symbol=ns.lookup(it->lhs.get("identifier"));

    //    out << symbol.hierarchy.size() << std::endl;

    if(symbol.hierarchy.size() > last_hierarchy.size())
    {
      symbolt::hierarchyt::const_iterator iter_hier = symbol.hierarchy.begin();
      symbolt::hierarchyt::const_iterator iter_hier_last = last_hierarchy.begin();

      unsigned pos = 0;

      for(; iter_hier_last != last_hierarchy.end(); iter_hier++, iter_hier_last++)
      {
        if(*iter_hier != *iter_hier_last)
          {
            //we found a mismatch in scopes
            //we have to put last_hierarchy.size() - pos $upscopes
            for(unsigned upscope = 0; upscope < last_hierarchy.size() - pos; upscope++)
              out << "$upscope " << " $end\n";
            break;
          }
        pos++;
      }

      for(; iter_hier != symbol.hierarchy.end(); iter_hier++)	
      {
        unsigned pos = id2string(*(iter_hier)).find_last_of(".");
        std::string change = id2string(*(iter_hier));
        out << "$scope module " << change.erase(0, pos+1) << " $end\n";	
      }

    }
    else if(symbol.hierarchy.size() <= last_hierarchy.size())
    {
      symbolt::hierarchyt::const_iterator iter_hier = symbol.hierarchy.begin();
      symbolt::hierarchyt::const_iterator iter_hier_last = last_hierarchy.begin();

      unsigned pos = 0;

      for(; iter_hier != symbol.hierarchy.end(); iter_hier++, iter_hier_last++)
      {
        if(*iter_hier != *iter_hier_last)
        {
          //we found a mismatch in scopes
          //we have to put last_hierarchy.size() - pos $upscopes
          for(unsigned upscope = 0; upscope < last_hierarchy.size() - pos; upscope++)
            out << "$upscope " << " $end\n";
          break;
        }

        pos++;
      }

      for(; iter_hier != symbol.hierarchy.end(); iter_hier++)	
      {
        unsigned pos = id2string(*(iter_hier)).find_last_of(".");
        std::string change = id2string(*(iter_hier));
        out << "$scope module " << change.erase(0, pos+1) << " $end\n";	
      }
    }

    std::string display_name = id2string(symbol.display_name());

    if(symbol.type.id()=="unsignedbv" ||
       symbol.type.id()=="signedbv" ||
       symbol.type.id()=="verilogbv")
    {
      unsigned left_bound, right_bound;

      std::string b = id2string(symbol.type.get("width"));
      std::string offset=id2string(symbol.type.get("#offset"));

      left_bound = 0+atoi(offset.c_str());
      right_bound = left_bound+atoi(b.c_str())-1;

      if(symbol.type.get_bool("#little_endian"))
        out << "$var reg " << symbol.type.get("width") << " " << vcd_identifier(display_name) << " " << 
          symbol.base_name << " [" << right_bound << ":" << left_bound << "]" << " $end\n";
      else
        out << "$var reg " << symbol.type.get("width") << " " << vcd_identifier(display_name) << " " << 
          symbol.base_name << " [" << left_bound << ":" << right_bound << "]" << " $end\n";
    }
    else
      out << "$var wire 1 " << vcd_identifier(display_name) << " " << symbol.base_name << "  $end\n";

    last_hierarchy = symbol.hierarchy;
  }
  
  //std::cout << "last scope " << scopes << std::endl;
  for(unsigned print = 0 ; print < last_hierarchy.size() ; print++)
    out << "$upscope $end\n";
    
  //  out << "$upscope $end\n";
  out << "$upscope $end\n";  

  out << "$enddefinitions $end\n";

  unsigned l=trace.get_failing_timeframe();

  for(unsigned t=0; t<=l; t++)
    show_trans_state_vcd(t, trace.states[t], ns, out);

  out << "$dumpall\n";
}

