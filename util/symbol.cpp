/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "symbol.h"
#include "location.h"

void symbolt::dump()
{
  show(std::cout);
}

void symbolt::show(std::ostream &out) const
{
  out << "  " << name << std::endl;
  out << "    type:  " << type.pretty(4) << std::endl
      << "    value: " << value.pretty(4) << std::endl;

  out << "  flags:";
  if(lvalue)          out << " lvalue";
  if(static_lifetime) out << " static_lifetime";
  if(file_local)      out << " file_local";
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
  out << std::endl;
}

std::ostream &operator<<(std::ostream &out,
                         const symbolt &symbol)
{
  symbol.show(out);
  return out;
}

void symbolt::to_irep(irept &dest) const
{
  dest.clear();
  dest.type()=type;
  dest.symvalue(value);
  dest.location(location);
  dest.name(name);
  dest.module(module);
  dest.base_name(base_name);
  dest.mode(mode);
  dest.pretty_name(pretty_name);
  dest.ordering(ordering);

  if (is_type) dest.is_type(true);
  if (is_macro) dest.is_macro(true);
  if (is_exported) dest.is_exported(true);
  if (is_input) dest.is_input(true);
  if (is_output) dest.is_output(true);
  if (is_statevar) dest.is_statevar(true);
  if (is_actual) dest.is_actual(true);
  if (free_var) dest.free_var(true);
  if (binding) dest.binding(true);
  if (lvalue) dest.lvalue(true);
  if (static_lifetime) dest.static_lifetime(true);
  if (file_local) dest.file_local(true);
  if (is_extern) dest.is_extern(true);
  if (is_volatile) dest.is_volatile(true);
}

void symbolt::from_irep(const irept &src)
{
  type=src.type();
  value=static_cast<const exprt &>(src.symvalue());
  location=static_cast<const locationt &>(src.location());

  name=src.name();
  module=src.module();
  base_name=src.base_name();
  mode=src.mode();
  pretty_name=src.pretty_name();
  ordering=atoi(src.ordering().c_str());

  is_type=src.is_type();
  is_macro=src.is_macro();
  is_exported=src.is_exported();
  is_input=src.is_input();
  is_output=src.is_output();
  is_statevar=src.is_statevar();
  is_actual=src.is_actual();
  free_var=src.free_var();
  binding=src.binding();
  lvalue=src.lvalue();
  static_lifetime=src.static_lifetime();
  file_local=src.file_local();
  is_extern=src.is_extern();
  is_volatile=src.is_volatile();
}
