/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "symbol.h"
#include "location.h"

void symbolt::dump() const
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
  if(is_macro)        out << " macro";
  if(is_parameter)    out << " parameter";
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

  if (is_type) dest.is_type(true);
  if (is_macro) dest.is_macro(true);
  if (is_parameter) dest.is_parameter(true);
  if (lvalue) dest.lvalue(true);
  if (static_lifetime) dest.static_lifetime(true);
  if (file_local) dest.file_local(true);
  if (is_extern) dest.is_extern(true);
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

  is_type=src.is_type();
  is_macro=src.is_macro();
  is_parameter=src.is_parameter();
  lvalue=src.lvalue();
  static_lifetime=src.static_lifetime();
  file_local=src.file_local();
  is_extern=src.is_extern();
}
