/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/location.h>
#include <util/symbol.h>

symbolt::symbolt()
{
  clear();
}

const irep_idt &symbolt::display_name() const
{
  return pretty_name.empty() ? name : pretty_name;
}

void symbolt::clear()
{
  value.make_nil();
  location.make_nil();
  lvalue=static_lifetime=file_local=is_extern=
    is_type=is_parameter=is_macro=is_used=false;
  name=module=base_name=mode=pretty_name="";
}

void symbolt::swap(symbolt &b)
{
#define SYM_SWAP1(x) x.swap(b.x)

  SYM_SWAP1(type);
  SYM_SWAP1(value);
  SYM_SWAP1(name);
  SYM_SWAP1(pretty_name);
  SYM_SWAP1(module);
  SYM_SWAP1(base_name);
  SYM_SWAP1(mode);
  SYM_SWAP1(location);

#define SYM_SWAP2(x) std::swap(x, b.x)

  SYM_SWAP2(is_type);
  SYM_SWAP2(is_macro);
  SYM_SWAP2(is_parameter);
  SYM_SWAP2(lvalue);
  SYM_SWAP2(static_lifetime);
  SYM_SWAP2(file_local);
  SYM_SWAP2(is_extern);
  SYM_SWAP2(is_used);
}

void symbolt::dump() const
{
  show(std::cout);
}

void symbolt::show(std::ostream &out) const
{
  out << "Symbol......: " << name << std::endl;
  out << "Pretty name.: " << pretty_name << std::endl;
  out << "Module......: " << module << std::endl;
  out << "Base name...: " << base_name << std::endl;
  out << "Mode........: " << mode << " (" << mode << ")" << std::endl;
  if(type.is_not_nil()) out << "Type........: " << type.pretty(4) << std::endl;
  if(value.is_not_nil()) out << "Value.......: " << value.pretty(4) <<  std::endl;

  out << "Flags.......:";

  if(lvalue)          out << " lvalue";
  if(static_lifetime) out << " static_lifetime";
  if(file_local)      out << " file_local";
  if(is_type)         out << " type";
  if(is_extern)       out << " extern";
  if(is_macro)        out << " macro";
  if(is_used)         out << " used";

  out << std::endl;
  out << "Location....: " << location << std::endl;

  out << std::endl;
}

std::ostream &operator<<(std::ostream &out, const symbolt &symbol)
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
