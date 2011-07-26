/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "symbol.h"
#include "location.h"

/*******************************************************************\

Function: symbolt::show

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
   
void symbolt::show(std::ostream &out) const
{
  out << "  " << name << std::endl;
  out << "    type:  " << type.pretty(4) << std::endl
      << "    value: " << value.pretty(4) << std::endl;

  out << "  flags:";
  if(lvalue)          out << " lvalue";
  if(static_lifetime) out << " static_lifetime";
  if(file_local)      out << " file_local";
  if(theorem)         out << " theorem";
  if(axiom)           out << " axiom";
  if(is_type)         out << " type";
  if(is_extern)       out << " extern";
  if(is_input)        out << " input";
  if(is_output)       out << " output";
  if(is_macro)        out << " macro";
  if(is_actual)       out << " actual";
  if(binding)         out << " binding";
  if(free_var)        out << " free_var";
  if(is_statevar)     out << " statevar";
  if(mode!="")        out << " mode=" << mode;
  if(base_name!="")   out << " base_name=" << base_name;
  if(module!="")      out << " module=" << module;
  if(pretty_name!="") out << " pretty_name=" << pretty_name;
  out << std::endl;
  out << "  location: " << location << std::endl;

  out << "  hierarchy:";

  for(std::list<irep_idt>::const_iterator it=hierarchy.begin();
      it!=hierarchy.end();
      it++)
    out << " " << *it;

  out << std::endl;  
  out << std::endl;  
}

/*******************************************************************\

Function: operator<<

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::ostream &operator<<(std::ostream &out,
                         const symbolt &symbol)
{
  symbol.show(out);
  return out;
}                        

/*******************************************************************\

Function: symbolt::to_irep

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symbolt::to_irep(irept &dest) const
{
  dest.clear();
  dest.type()=type;
  dest.add("symvalue")=value;
  dest.add("location")=location;
  dest.set("name", name);
  dest.set("module", module);
  dest.set("base_name", base_name);
  dest.set("mode", mode);
  dest.set("pretty_name", pretty_name);
  dest.set("ordering", ordering);

  if (theorem) dest.set("theorem", true);
  if (axiom) dest.set("axiom", true);
  if (is_type) dest.set("is_type", true);
  if (is_macro) dest.set("is_macro", true);
  if (is_exported) dest.set("is_exported", true);
  if (is_input) dest.set("is_input", true);
  if (is_output) dest.set("is_output", true);
  if (is_statevar) dest.set("is_statevar", true);
  if (is_actual) dest.set("is_actual", true);
  if (free_var) dest.set("free_var", true);
  if (binding) dest.set("binding", true);
  if (lvalue) dest.set("lvalue", true);
  if (static_lifetime) dest.set("static_lifetime", true);
  if (file_local) dest.set("file_local", true);
  if (is_extern) dest.set("is_extern", true);
  if (is_volatile) dest.set("is_volatile", true);       
}

/*******************************************************************\

Function: symbolt::from_irep

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void symbolt::from_irep(const irept &src)
{
  type=src.type();
  value=static_cast<const exprt &>(src.find("symvalue"));
  location=static_cast<const locationt &>(src.find("location"));

  name=src.name();
  module=src.get("module");
  base_name=src.get("base_name");
  mode=src.get("mode");
  pretty_name=src.get("pretty_name");
  ordering=atoi(src.get("ordering").c_str());

  theorem=src.get_bool("theorem");
  axiom=src.get_bool("axiom");
  is_type=src.get_bool("is_type");
  is_macro=src.get_bool("is_macro");
  is_exported=src.get_bool("is_exported");
  is_input=src.get_bool("is_input");
  is_output=src.get_bool("is_output");
  is_statevar=src.get_bool("is_statevar");
  is_actual=src.get_bool("is_actual");
  free_var=src.get_bool("free_var");
  binding=src.get_bool("binding");
  lvalue=src.get_bool("lvalue");
  static_lifetime=src.get_bool("static_lifetime");
  file_local=src.get_bool("file_local");
  is_extern=src.get_bool("is_extern");
  is_volatile=src.get_bool("is_volatile");
}
