/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <location.h>
#include <simplify_expr_class.h>
#include <ansi-c/c_qualifiers.h>

#include "cpp_typecheck.h"
#include "cpp_convert_type.h"
#include "expr2cpp.h"

/*******************************************************************\

Function: cpp_typecheckt::typecheck_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/
void cpp_typecheckt::typecheck_type(typet &type)
{
  assert(type.id()!="");
  assert(type.is_not_nil());

  try
  {
    cpp_convert_plain_type(type);
  }

  catch(const char *error)
  {
    err_location(type);
    str << error << std::endl;
    throw 0;
  }

  catch(const std::string &error)
  {
    err_location(type);
    str << error << std::endl;
    throw 0;
  }

  if(type.id()=="cpp-name")
  {
    c_qualifierst qualifiers(type);
    cpp_namet cpp_name;
    cpp_name.swap(type);
    exprt symbol_expr;
    resolve(
      cpp_name,
      cpp_typecheck_resolvet::TYPE,
      cpp_typecheck_fargst(),
      symbol_expr);

    if(symbol_expr.id()!="type")
    {
      err_location(type);
      str << "error: expected type" << std::endl;
      throw 0;
    }
    type = symbol_expr.type();

    if(type.get_bool("#constant"))
      qualifiers.is_constant = true;

     qualifiers.write(type);
  }
  else if(type.id()=="struct" ||
          type.id()=="union")
  {
    typecheck_compound_type(type);
  }
  else if(type.id()=="pointer")
  {
    // the pointer might have a qualifier, but do subtype first
    typecheck_type(type.subtype());

    // Check if it is a pointer-to-member
    if(type.find("to-member").is_not_nil())
    {
      // Must point to a method
      if(type.subtype().id()!= "code")
      {
        err_location(type);
        str << "pointer-to-member musst point to a method: "
            << type.subtype().pretty() << std::endl;
        throw 0;
      }

      typet &member=static_cast<typet &>(type.add("to-member"));

      if(member.id()=="cpp-name")
      {
        assert(member.get_sub().back().id()=="::");
        member.get_sub().pop_back();
      }

      typecheck_type(member);

      irept::subt &args=type.subtype().add("arguments").get_sub();
      if(args.empty() || args.front().get("#base_name")!="this")
      {
        // Add 'this' to the arguments
        exprt a0("argument");
        a0.set("#base_name","this");
        a0.type().id("pointer");
        a0.type().subtype() = member;
        args.insert(args.begin(),a0);
      }
    }

    // now do qualifier
    if(type.find("#qualifier").is_not_nil())
    {
      typet &t=(typet &)type.add("#qualifier");
      cpp_convert_plain_type(t);
      c_qualifierst q(t);
      q.write(type);
    }

    type.remove("#qualifier");
  }
  else if(type.id()=="array")
  {
    exprt &size_expr=static_cast<exprt &>(type.add("size"));

    if(size_expr.is_nil())
      type.id("incomplete_array");
    else
    {
      typecheck_expr(size_expr);

      simplify_exprt expr_simplifier;
      expr_simplifier.simplify(size_expr);

      if(size_expr.id() != "constant" &&
         size_expr.id() != "infinity")
      {
        err_location(type);
        str << "failed to determine size of array: " <<
          expr2cpp(size_expr,context) << std::endl;
        throw 0;
      }
    }

    typecheck_type(type.subtype());

    if(type.subtype().get_bool("#constant"))
      type.set("#constant", true);

    if(type.subtype().get_bool("#volatile"))
      type.set("#volatile", true);
  }
  else if(type.id()=="code")
  {
    code_typet &code_type=to_code_type(type);
    typecheck_type(code_type.return_type());

    code_typet::argumentst &arguments=code_type.arguments();

    for(code_typet::argumentst::iterator it=arguments.begin();
        it!=arguments.end();
        it++)
    {
      typecheck_type(it->type());

      // see if there is a default value
      if(it->has_default_value())
      {
        typecheck_expr(it->default_value());
        implicit_typecast(it->default_value(), it->type());
      }
    }
  }
  else if(type.id()=="template")
  {
    typecheck_type(type.subtype());
  }
  else if(type.id()=="c_enum")
  {
    typecheck_enum_type(type);
  }
  else if(type.id()=="unsignedbv" ||
          type.id()=="signedbv" ||
          type.id()=="bool" ||
          type.id()=="floatbv" ||
          type.id()=="fixedbv" ||
          type.id()=="empty")
  {
  }
  else if(type.id()=="symbol")
  {
  }
  else if(type.id()=="constructor" ||
          type.id()=="destructor")
  {
  }
  else if(type.id()=="cpp-cast-operator")
  {
  }
  else if(type.id()=="cpp-template-type")
  {
  }
  #ifdef CPP_SYSTEMC_EXTENSION
  else if(type.id() == "verilogbv")
  {
  }
  #endif
  else
  {
    err_location(type);
    str << "unexpected type: " << type.pretty() << std::endl;
    throw 0;
  }
}
