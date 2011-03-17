/*******************************************************************\

Module: C++ Language Type Conversion

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <assert.h>

#include <iostream>

#include <config.h>
#include <arith_tools.h>
#include <std_types.h>

#include <ansi-c/c_types.h>

#include "cpp_convert_type.h"
#include "cpp_declaration.h"
#include "cpp_name.h"

class cpp_convert_typet
{
public:
  unsigned unsigned_cnt, signed_cnt, char_cnt, int_cnt, short_cnt,
           long_cnt, const_cnt, typedef_cnt, volatile_cnt,
           double_cnt, float_cnt, bool_cnt, extern_cnt, wchar_t_cnt,
           int8_cnt, int16_cnt, int32_cnt, int64_cnt, ptr32_cnt, ptr64_cnt;

  void read(const typet &type);
  void write(typet &type);

  std::list<typet> other;

  cpp_convert_typet() { }
  cpp_convert_typet(const typet &type) { read(type); }

protected:
  void read_rec(const typet &type);
  void read_function_type(const typet &type);
  void read_template(const typet &type);
};

/*******************************************************************\

Function: cpp_convert_typet::read

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_convert_typet::read(const typet &type)
{
  unsigned_cnt=signed_cnt=char_cnt=int_cnt=short_cnt=
  long_cnt=const_cnt=typedef_cnt=volatile_cnt=
  double_cnt=float_cnt=bool_cnt=extern_cnt=
  wchar_t_cnt=int8_cnt=int16_cnt=int32_cnt=
  int64_cnt=ptr32_cnt=ptr64_cnt=0;

  other.clear();

  read_rec(type);
}

/*******************************************************************\

Function: cpp_convert_typet::read_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_convert_typet::read_rec(const typet &type)
{
  #if 0
  std::cout << "cpp_convert_typet::read_rec: "
            << type.pretty() << std::endl;
  #endif

  if(type.id()=="merged_type")
  {
    forall_subtypes(it, type)
      read_rec(*it);
  }
  else if(type.id()=="signed")
    signed_cnt++;
  else if(type.id()=="unsigned")
    unsigned_cnt++;
  else if(type.id()=="volatile")
    volatile_cnt++;
  else if(type.id()=="char")
    char_cnt++;
  else if(type.id()=="int")
    int_cnt++;
  else if(type.id()=="short")
    short_cnt++;
  else if(type.id()=="long")
    long_cnt++;
  else if(type.id()=="double")
    double_cnt++;
  else if(type.id()=="float")
    float_cnt++;
  else if(type.id()=="bool")
    bool_cnt++;
  else if(type.id()=="wchar_t")
    wchar_t_cnt++;
  else if(type.id()=="__int8")
    int8_cnt++;
  else if(type.id()=="__int16")
    int16_cnt++;
  else if(type.id()=="__int32")
    int32_cnt++;
  else if(type.id()=="__int64")
    int64_cnt++;
  else if(type.id()=="__ptr32")
    ptr32_cnt++;
  else if(type.id()=="__ptr64")
    ptr64_cnt++;
  else if(type.id()=="const")
    const_cnt++;
  else if(type.id()=="extern")
    extern_cnt++;
  else if(type.id()=="function_type")
  {
    read_function_type(type);
  }
  else if(type.id()=="typedef")
    typedef_cnt++;
  else if(type.id()=="identifier")
  {
    // from arguments
  }
  else if(type.id()=="cpp-name")
  {
    // from typedefs
    other.push_back(type);
  }
  else if(type.id()=="array")
  {
    other.push_back(type);
    cpp_convert_plain_type(other.back().subtype());
  }
  else if(type.id()=="template")
  {
    read_template(type);
  }
  else
  {
    other.push_back(type);
  }
}

/*******************************************************************\

Function: cpp_covnert_typet::read_template

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_convert_typet::read_template(const typet &type)
{
  other.push_back(type);
  typet &t=other.back();

  cpp_convert_plain_type(t.subtype());

  irept &arguments=t.add("arguments");

  forall_irep(it, arguments.get_sub())
  {
    exprt &decl=(exprt &)*it;

    // may be type or expression
    bool is_type=decl.get_bool("is_type");

    if(is_type)
    {
    }
    else
    {
      cpp_convert_plain_type(decl.type());
    }

    // TODO: initializer
  }
}

/*******************************************************************\

Function: cpp_convert_typet::read_function_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_convert_typet::read_function_type(const typet &type)
{
  other.push_back(type);
  typet &t=other.back();
  t.id("code");

  // change subtype to return_type
  typet &return_type=
    static_cast<typet &>(t.add("return_type"));

  return_type.swap(t.subtype());
  t.remove("subtype");

  if(return_type.is_not_nil())
    cpp_convert_plain_type(return_type);

  // take care of argument types
  irept &arguments=t.add("arguments");

  // see if we have an ellipsis
  if(!arguments.get_sub().empty() &&
     arguments.get_sub().back().id()=="ellipsis")
  {
    arguments.set("ellipsis", true);
    arguments.get_sub().erase(--arguments.get_sub().end());
  }

  Forall_irep(it, arguments.get_sub())
  {
    exprt &argument_expr=static_cast<exprt &>(*it);

    if(argument_expr.id()=="cpp-declaration")
    {
      cpp_declarationt &declaration=to_cpp_declaration(argument_expr);
      locationt type_location=declaration.type().location();

      cpp_convert_plain_type(declaration.type());

      // there should be only one declarator
      assert(declaration.declarators().size()==1);

      cpp_declaratort &declarator=
        declaration.declarators().front();

      // do we have a declarator?
      if(declarator.is_nil())
      {
        argument_expr=exprt("argument", declaration.type());
        argument_expr.location()=type_location;
      }
      else
      {
        const cpp_namet &cpp_name=declarator.name();
        typet final_type=declarator.convert(declaration.type());

        // see if it's an array type
        if(final_type.id()=="array" || final_type.id()=="incomplete_array")
        {
          final_type.id("pointer");
          final_type.remove("size");
        }

        code_typet::argumentt new_argument(final_type);

        if(cpp_name.is_nil())
        {
          new_argument.location()=type_location;
        }
        else
        {
          std::string identifier, base_name;
          cpp_name.convert(identifier, base_name);
          assert(!identifier.empty());
          new_argument.set_identifier(identifier);
          new_argument.set_base_name(identifier);
          new_argument.location()=cpp_name.location();
        }

        if(declarator.value().is_not_nil())
          new_argument.default_value().swap(declarator.value());

        argument_expr.swap(new_argument);
      }
    }
    else if(argument_expr.id()=="ellipsis")
      throw "ellipsis only allowed as last argument";
    else
      assert(0);
  }

  // if we just have one argument of type void, remove it
  if(arguments.get_sub().size()==1 &&
     arguments.get_sub().front().find("type").id()=="empty")
    arguments.get_sub().clear();
}

/*******************************************************************\

Function: cpp_convert_typet::write

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_convert_typet::write(typet &type)
{
  type.clear();

  // first, do "other"

  if(!other.empty())
  {
    if(double_cnt || float_cnt || signed_cnt ||
       unsigned_cnt || int_cnt || bool_cnt ||
       short_cnt || char_cnt || wchar_t_cnt ||
       int8_cnt || int16_cnt || int32_cnt ||
       int64_cnt || ptr32_cnt || ptr64_cnt)
      throw "illegal type modifier";

    if(other.size()!=1)
      throw "illegal combination of types";

    type.swap(other.front());
  }
  else if(double_cnt || float_cnt)
  {
    if(signed_cnt || unsigned_cnt || int_cnt || bool_cnt ||
       short_cnt || char_cnt || wchar_t_cnt ||
       int8_cnt || int16_cnt || int32_cnt ||
       int64_cnt || ptr32_cnt || ptr64_cnt)
      throw "illegal type modifier";

    if(double_cnt && float_cnt)
      throw "cannot use both float and double";

    if(float_cnt)
    {
      type = float_type();
    }
    else
    {
      type = double_type();
    }
  }
  else if(bool_cnt)
  {
    if(signed_cnt || unsigned_cnt || int_cnt || short_cnt ||
       char_cnt || wchar_t_cnt ||
       int8_cnt || int16_cnt || int32_cnt ||
       int64_cnt || ptr32_cnt || ptr64_cnt)
      throw "illegal type modifier";

    type.id("bool");
  }
  else if(ptr32_cnt)
  {
    type.id("pointer");
    type.subtype().id("empty");
  }
  else if(ptr64_cnt)
  {
    type.id("pointer");
    type.subtype().id("empty");
  }
  else
  {
    // it is integer -- signed or unsigned?

    if(signed_cnt && unsigned_cnt)
      throw "illegal type modifier";
    else if(unsigned_cnt)
      type.id("unsignedbv");
    else if(signed_cnt)
      type.id("signedbv");
    else
    {
      if(char_cnt)
        type.id(config.ansi_c.char_is_unsigned?"unsignedbv":"signedbv");
      else if(wchar_t_cnt)
        type.id("signedbv");
      else
        type.id("signedbv");
    }

    // get width

    unsigned width;

    if(short_cnt)
    {
      if(long_cnt || char_cnt)
        throw "illegal type modifier";

      width=config.ansi_c.short_int_width;
    }
    else if(char_cnt)
    {
      if(long_cnt)
        throw "illegal type modifier";

      width=config.ansi_c.char_width;
    }
    else if(wchar_t_cnt)
    {
      if(long_cnt)
        throw "illegal type modifier";

      width=config.ansi_c.wchar_t_width;
    }
    else if(int8_cnt)
    {
      if(long_cnt)
        throw "illegal type modifier";

      width=1;
    }
    else if(int16_cnt)
    {
      if(long_cnt)
        throw "illegal type modifier";

      width=2;
    }
    else if(int32_cnt)
    {
      if(long_cnt)
        throw "illegal type modifier";

      width=4;
    }
    else if(int64_cnt)
    {
      if(long_cnt)
        throw "illegal type modifier";

      width=8;
    }
    else if(!long_cnt)
    {
      width=config.ansi_c.int_width;
    }
    else if(long_cnt==1)
    {
      width=config.ansi_c.long_int_width;
    }
    else if(long_cnt==2)
    {
      width=config.ansi_c.long_long_int_width;
    }
    else
      throw "illegal type modifier";

    type.set("width", width);
  }

  // is it constant?
  if(const_cnt)
    type.set("#constant", true);

  // is it volatile?
  if(volatile_cnt)
    type.set("#volatile", true);
}

/*******************************************************************\

Function: cpp_convert_plain_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_convert_plain_type(typet &type)
{
  if(type.id()=="cpp-name" ||
     type.id()=="struct" ||
     type.id()=="union" ||
     type.id()=="pointer" ||
     type.id()=="array" ||
     type.id()=="code" ||
     type.id()=="unsignedbv" ||
     type.id()=="signedbv" ||
     type.id()=="bool" ||
     type.id()=="floatbv" ||
     type.id()=="empty" ||
     type.id()=="symbol" ||
     type.id()=="c_enum" ||
     type.id()=="constructor" ||
     type.id()=="destructor")
  {
  }
  else
  {
    cpp_convert_typet cpp_convert_type(type);
    cpp_convert_type.write(type);
  }
}
