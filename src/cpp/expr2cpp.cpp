/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cassert>
#include <clang-c-frontend/expr2c.h>
#include <cpp/expr2cpp.h>
#include <util/std_types.h>
#include <util/symbol.h>

class expr2cppt:public expr2ct
{
public:
  expr2cppt(const namespacet &_ns, const bool _fullname) : expr2ct(_ns, _fullname) { }

  std::string convert(const exprt &src) override
  {
    return expr2ct::convert(src);
  }

  std::string convert(const typet &src) override
  {
    return expr2ct::convert(src);
  }

protected:
  std::string convert(const exprt &src, unsigned &precedence) override;
  virtual std::string convert_cpp_this(const exprt &src, unsigned precedence);
  virtual std::string convert_cpp_new(const exprt &src, unsigned precedence);
  virtual std::string convert_code_cpp_delete(const exprt &src, unsigned precedence);
  std::string convert_struct(const exprt &src, unsigned &precedence) override;
  std::string convert_code(const codet &src, unsigned indent) override;
  std::string convert_constant(const exprt &src, unsigned &precedence) override;

  std::string convert_rec(
    const typet &src,
    const c_qualifierst &qualifiers,
    const std::string &declarator) override;

  typedef hash_set_cont<std::string, string_hash> id_sett;
};

std::string expr2cppt::convert_struct(
  const exprt &src,
  unsigned &precedence)
{
  const typet &full_type=ns.follow(src.type());

  if(full_type.id()!="struct")
    return convert_norep(src, precedence);

  const struct_typet &struct_type=to_struct_type(full_type);

  std::string dest="{ ";

  const struct_typet::componentst &components=
    struct_type.components();

  assert(components.size()==src.operands().size());

  exprt::operandst::const_iterator o_it=src.operands().begin();

  bool first=true;
  unsigned last_size=0;

  for(const auto & component : components)
  {
    if(component.type().id()=="code")
    {
    }
    else
    {
      std::string tmp=convert(*o_it);
      std::string sep;

      if(first)
        first=false;
      else
      {
        if(last_size+40<dest.size())
        {
          sep=",\n    ";
          last_size=dest.size();
        }
        else
          sep=", ";
      }

      dest+=sep;
      dest+=".";
      dest+=component.get_string("pretty_name");
      dest+="=";
      dest+=tmp;
    }

    o_it++;
  }

  dest+=" }";

  return dest;
}

std::string expr2cppt::convert_constant(
  const exprt &src,
  unsigned &precedence)
{
  if(src.type().id()=="bool")
  {
    // C++ has built-in Boolean constants, in contrast to C
    if(src.is_true())
      return "true";
    else if(src.is_false())
      return "false";
  }

  return expr2ct::convert_constant(src, precedence);
}

std::string expr2cppt::convert_rec(
  const typet &src,
  const c_qualifierst &qualifiers,
  const std::string &declarator)
{
  c_qualifierst new_qualifiers(qualifiers);
  new_qualifiers.read(src);

  const std::string d=
    declarator==""?declarator:(" "+declarator);

  const std::string q=
    new_qualifiers.as_string();

  if(is_reference(src))
  {
    return new_qualifiers.as_string()+convert(src.subtype())+" &"+d;
  }
  else if(is_rvalue_reference(src))
  {
    return new_qualifiers.as_string()+convert(src.subtype())+" &&"+d;
  }
  else if(src.get("#cpp_type")!="")
  {
    const irep_idt cpp_type=src.get("#cpp_type");

    if(cpp_type=="signed_char")
      return new_qualifiers.as_string()+"signed char"+d;
    else if(cpp_type=="unsigned_char")
      return new_qualifiers.as_string()+"unsigned char"+d;
    else if(cpp_type=="char")
      return new_qualifiers.as_string()+"char"+d;
    else if(cpp_type=="signed_short_int")
      return new_qualifiers.as_string()+"short"+d;
    else if(cpp_type=="signed_short_int")
      return new_qualifiers.as_string()+"unsigned short"+d;
    else if(cpp_type=="signed_int")
      return new_qualifiers.as_string()+"int"+d;
    else if(cpp_type=="unsigned_int")
      return new_qualifiers.as_string()+"unsigned"+d;
    else if(cpp_type=="signed_long_int")
      return new_qualifiers.as_string()+"long"+d;
    else if(cpp_type=="unsigned_long_int")
      return new_qualifiers.as_string()+"unsigned long"+d;
    else if(cpp_type=="signed_long_long_int")
      return new_qualifiers.as_string()+"long long"+d;
    else if(cpp_type=="unsigned_long_long_int")
      return new_qualifiers.as_string()+"unsigned long long"+d;
    else if(cpp_type=="wchar_t")
      return new_qualifiers.as_string()+"wchar_t"+d;
    else if(cpp_type=="float")
      return new_qualifiers.as_string()+"float"+d;
    else if(cpp_type=="double")
      return new_qualifiers.as_string()+"double"+d;
    else if(cpp_type=="long_double")
      return new_qualifiers.as_string()+"long double"+d;
    else
      return expr2ct::convert_rec(src, qualifiers, declarator);
  }
  else if(src.id()=="symbol")
  {
    const irep_idt &identifier=src.identifier();

    const symbolt &symbol=ns.lookup(identifier);

    if(symbol.type.id()=="struct" ||
       symbol.type.id()=="incomplete_struct")
    {
      std::string dest=new_qualifiers.as_string();

      if(symbol.type.get_bool("#class"))
        dest+="class";
      else if(symbol.type.get_bool("#interface"))
        dest+="__interface"; // MS-specific
      else
        dest+="struct";

      if(symbol.pretty_name!=irep_idt())
        dest+=" "+id2string(symbol.pretty_name);

      dest+=d;

      return dest;
    }
    else if(symbol.type.id()=="c_enum")
    {
      std::string dest=new_qualifiers.as_string();

      dest+="enum";

      if(symbol.pretty_name!=irep_idt())
        dest+=" "+id2string(symbol.pretty_name);

      dest+=d;

      return dest;
    }
    else
      return expr2ct::convert_rec(src, qualifiers, declarator);
  }
  else if(src.id()=="struct" ||
          src.id()=="incomplete_struct")
  {
    std::string dest=new_qualifiers.as_string();

    if(src.get_bool("#class"))
      dest+="class";
    else if(src.get_bool("#interface"))
      dest+="__interface"; // MS-specific
    else
      dest+="struct";

    dest+=d;

    return dest;
  }
  else if(src.id()=="constructor")
  {
    return "constructor ";
  }
  else if(src.id()=="destructor")
  {
    return "destructor ";
  }
  else if(src.id()=="cpp-template-type")
  {
    return "typename";
  }
  else if(src.id()=="template")
  {
    std::string dest="template<";

    const irept::subt &arguments=src.arguments().get_sub();

    forall_irep(it, arguments)
    {
      if(it!=arguments.begin()) dest+=", ";

      const exprt &argument=(const exprt &)*it;

      if(argument.id()=="symbol")
      {
        dest+=convert(argument.type())+" ";
        dest+=convert(argument);
      }
      else if(argument.id()=="type")
        dest+=convert(argument.type());
      else
        dest+=argument.to_string();
    }

    dest+="> "+convert(src.subtype());
    return dest;
  }
  else if(src.id()=="pointer" && src.find("to-member").is_not_nil())
  {
    typet tmp=src;
    typet member;
    member.swap(tmp.add("to-member"));

    std::string dest = "(" + convert_rec(member, c_qualifierst(), "") + ":: *)";

    if(src.subtype().id()=="code")
    {
      const code_typet& code_type = to_code_type(src.subtype());
      const typet& return_type = code_type.return_type();
      dest = convert_rec(return_type, c_qualifierst(), "") +" " + dest;

      const code_typet::argumentst& args = code_type.arguments();
      dest += "(";

      if(args.size() > 0)
        dest += convert_rec(args[0].type(), c_qualifierst(), "");

      for(unsigned i = 1; i < args.size();i++)
        dest += ", " + convert_rec(args[i].type(), c_qualifierst(), "");
      dest += ")";
      dest+=d;
    }
    else
    {
      dest = convert_rec(src.subtype(),c_qualifierst(), "") + " " + dest+d;
    }
    return dest;
  }
  else if(src.id()=="unassigned")
    return "?";
  else
    return expr2ct::convert_rec(src, qualifiers, declarator);
}

std::string expr2cppt::convert_cpp_this(
  const exprt &src __attribute__((unused)),
  unsigned precedence __attribute__((unused)))
{
  return "this";
}

std::string expr2cppt::convert_cpp_new(
  const exprt &src,
  unsigned precedence __attribute__((unused)))
{
  std::string dest;

  if(src.statement()=="cpp_new[]")
  {
    dest="new";

    std::string tmp_size=
      convert(static_cast<const exprt &>(src.size_irep()));

    dest+=" ";
    dest+=convert(src.type().subtype());
    dest+="[";
    dest+=tmp_size;
    dest+="]";
  }
  else
    dest="new "+convert(src.type().subtype());

  return dest;
}

std::string expr2cppt::convert_code_cpp_delete(
  const exprt &src,
  unsigned indent)
{
  std::string dest=indent_str(indent)+"delete ";

  if(src.operands().size()!=1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  std::string tmp=convert(src.op0());

  dest+=tmp+";\n";

  return dest;
}

std::string expr2cppt::convert(
  const exprt &src,
  unsigned &precedence)
{
  if(src.id()=="cpp-this")
    return convert_cpp_this(src, precedence=15);
  else if(src.id()=="sideeffect" &&
          (src.statement()=="cpp_new" ||
           src.statement()=="cpp_new[]"))
    return convert_cpp_new(src, precedence=15);
  else if(src.id()=="unassigned")
    return "?";
  else if(src.id()=="pod_constructor")
    return "pod_constructor";
  else
    return expr2ct::convert(src, precedence);
}

std::string expr2cppt::convert_code(
  const codet &src,
  unsigned indent)
{
  const irep_idt &statement=src.statement();

  if(statement=="cpp_delete" ||
     statement=="cpp_delete[]")
    return convert_code_cpp_delete(src, indent);

  if(statement=="cpp_new" ||
     statement=="cpp_new[]")
    return convert_cpp_new(src,indent);

  return expr2ct::convert_code(src, indent);
}

std::string expr2cpp(const exprt &expr, const namespacet &ns, const bool fullname)
{
  expr2cppt expr2cpp(ns, fullname);
  expr2cpp.get_shorthands(expr);
  return expr2cpp.convert(expr);
}

std::string type2cpp(const typet &type, const namespacet &ns, const bool fullname)
{
  expr2cppt expr2cpp(ns, fullname);
  return expr2cpp.convert(type);
}
