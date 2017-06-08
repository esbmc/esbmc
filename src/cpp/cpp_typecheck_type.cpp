/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <util/c_qualifiers.h>
#include <cpp/cpp_convert_type.h>
#include <cpp/cpp_typecheck.h>
#include <cpp/expr2cpp.h>
#include <util/location.h>

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
    str << error;
    throw 0;
  }

  catch(const std::string &error)
  {
    err_location(type);
    str << error;
    throw 0;
  }

  if(type.id()=="cpp-name")
  {
    c_qualifierst qualifiers(type);

    cpp_namet cpp_name;
    cpp_name.swap(type);

    exprt symbol_expr=resolve(
      cpp_name,
      cpp_typecheck_resolvet::TYPE,
      cpp_typecheck_fargst());

    if(symbol_expr.id()!="type")
    {
      err_location(type);
      str << "error: expected type";
      throw 0;
    }

    type=symbol_expr.type();
    assert(type.is_not_nil());

    if(type.cmt_constant())
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
      // these can point either to data members or member functions
      // of a class

      typet &class_object=static_cast<typet &>(type.add("to-member"));

      if(class_object.id()=="cpp-name")
      {
        assert(class_object.get_sub().back().id()=="::");
        class_object.get_sub().pop_back();
      }

      typecheck_type(class_object);

      // there may be arguments if this is a pointer to member function
      if(type.subtype().id()=="code")
      {
        irept::subt &args=type.subtype().add("arguments").get_sub();

        if(args.empty() ||
           args.front().cmt_base_name()!="this")
        {
          // Add 'this' to the arguments
          exprt a0("argument");
          a0.cmt_base_name("this");
          a0.type().id("pointer");
          a0.type().subtype() = class_object;
          args.insert(args.begin(),a0);
        }
      }
    }

    // now do qualifier
    if(type.find("#qualifier").is_not_nil())
    {
      typet &t=static_cast<typet &>(type.add("#qualifier"));
      cpp_convert_plain_type(t);
      c_qualifierst q(t);
      q.write(type);
    }

    type.remove("#qualifier");
  }
  else if(type.id()=="array")
  {
    exprt &size_expr=to_array_type(type).size();

    if(size_expr.is_nil())
      type.id("incomplete_array");
    else
      typecheck_expr(size_expr);

    // TODO: If is a incomplete_array, it should always
    // have initializers, except for catch declaration

    typecheck_type(type.subtype());

    if(type.subtype().cmt_constant())
      type.cmt_constant(true);

    if(type.subtype().cmt_volatile())
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
  else if(type.id()=="typeof")
  {
    exprt e=static_cast<const exprt &>(type.find("expr"));

    if(e.is_nil())
    {
      typet tmp_type=
        static_cast<const typet &>(type.find("sizeof-type"));

      if(tmp_type.id()=="cpp-name")
      {
        // this may be ambiguous -- it can be either a type or
        // an expression

        cpp_typecheck_fargst fargs;

        exprt symbol_expr=resolve(
          to_cpp_name(static_cast<const irept &>(tmp_type)),
          cpp_typecheck_resolvet::BOTH,
          fargs);

        type=symbol_expr.type();
      }
      else
      {
        typecheck_type(tmp_type);
        type=tmp_type;
      }
    }
    else
    {
      typecheck_expr(e);
      type=e.type();
    }
  }
  else if(type.id()=="decltype")
  {
    exprt e=static_cast<const exprt &>(type.find("expr_arg"));
    typecheck_expr(e);
    type=e.type();
  }
  else if(type.id()=="unassigned")
  {
    // ignore, for template argument guessing
  }
  else if(type.id()=="ellipsis")
  {
  }
  else
  {
    err_location(type);
    str << "unexpected type: " << type.pretty();
    throw 0;
  }

  assert(type.is_not_nil());
}
