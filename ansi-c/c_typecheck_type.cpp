/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <config.h>
#include <simplify_expr.h>
#include <arith_tools.h>
#include <std_types.h>
#include <i2string.h>
#include <expr_util.h>

#include "c_typecheck_base.h"
#include "c_types.h"

/*******************************************************************\

Function: c_typecheck_baset::typecheck_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_type(typet &type)
{
  if(type.is_code())
  {
    code_typet &code_type=to_code_type(type);

    code_typet::argumentst &arguments=code_type.arguments();

    // if we don't have any arguments, we assume it's (...)
    if(arguments.empty())
    {
      code_type.make_ellipsis();
    }
    else if(arguments.size()==1 &&
            arguments[0].type().is_empty())
    {
      // if we just have one argument of type void, remove it
      arguments.clear();
    }
    else
    {
      for(unsigned i=0; i<code_type.arguments().size(); i++)
      {
        code_typet::argumentt &argument=code_type.arguments()[i];
        typet &type=argument.type();

        if(type.id()=="KnR")
        {
          // need to look it up
          irep_idt identifier=argument.get_identifier();

          if(identifier=="")
          {
            err_location(argument);
            throw "failed to find identifier for K&R function argument";
          }

          // may be renamed
          {
            id_replace_mapt::const_iterator m_it=id_replace_map.find(identifier);
            if(m_it!=id_replace_map.end())
              identifier=m_it->second;
          }

          symbolst::iterator s_it=context.symbols.find(identifier);
          if(s_it==context.symbols.end())
          {
            err_location(argument);
            throw "failed to find K&R function argument symbol";
          }
          
          symbolt &symbol=s_it->second;
          
          if(symbol.type.id()=="KnR")
          {
            err_location(argument);
            throw "failed to get a type for K&R function argument";
          }

          adjust_function_argument(symbol.type);
          type=symbol.type;
        }
        else
        {
          typecheck_type(type);
          adjust_function_argument(type);
        }
      }
    }

    typecheck_type(code_type.return_type());
  }
  else if(type.is_array())
  {
    array_typet &array_type=to_array_type(type);
    exprt &size=array_type.size();

    typecheck_expr(size);
    typecheck_type(array_type.subtype());

    bool size_is_unsigned=(size.type().is_unsignedbv());

    typet integer_type(size_is_unsigned?"unsignedbv":"signedbv");
    integer_type.width(config.ansi_c.int_width);

    implicit_typecast(size, integer_type);

    // simplify it
    simplify(size);

    if(size.is_constant())
    {
      mp_integer s;
      if(to_integer(size, s))
      {
        err_location(size);
        str << "failed to convert constant: "
            << size.pretty();
        throw 0;
      }

      if(s<0)
      {
        err_location(size);
        str << "array size must not be negative, "
               "but got " << s;
        throw 0;
      }
    }
  }
  else if(type.is_incomplete_array())
  {
    typecheck_type(type.subtype());
  }
  else if(type.is_pointer())
  {
    typecheck_type(type.subtype());
  }
  else if(type.is_struct() ||
          type.is_union())
  {
    struct_typet &struct_type=to_struct_type(type);
    struct_typet::componentst &components=struct_type.components();

    for(struct_typet::componentst::iterator
        it=components.begin();
        it!=components.end();
        it++)
    {
      typet &type=it->type();
    
      typecheck_type(type);
      
      // incomplete arrays become arrays of size 0
      if(type.is_incomplete_array())
      {
        type.id("array");
        type.size(gen_zero(int_type()));
      }
    }

    unsigned anon_member_counter=0;

    // scan for anonymous members
    for(struct_typet::componentst::iterator
        it=components.begin();
        it!=components.end();
        ) // no it++
    {
      if(it->name()=="")
      {
        const typet &final_type=follow(it->type());

        if(final_type.is_struct() ||
           final_type.is_union())
        {
          struct_typet::componentt c;
          c.swap(*it);
          it=components.erase(it);

          // copy child's records
          const typet &final_type=follow(c.type());
          if(!final_type.is_struct() &&
             !final_type.is_union())
          {
            err_location(type);
            str << "expected struct or union as anonymous member, but got `"
                << to_string(final_type) << "'";
            throw 0;
          }

          const struct_typet &c_struct_type=to_struct_type(final_type);
          const struct_typet::componentst &c_components=
            c_struct_type.components();

          for(struct_typet::componentst::const_iterator
              c_it=c_components.begin();
              c_it!=c_components.end();
              c_it++)
            it=components.insert(it, *c_it);
        }
        else
        {
          // some other anonymous member
          it->name("$anon_member"+i2string(anon_member_counter++));
          it++;
        }
      }
      else
        it++;
    }
  }
  else if(type.is_c_enum())
  {
  }
  else if(type.id()=="c_bitfield")
  {
    typecheck_type(type.subtype());

    // we turn this into unsigedbv/signedbv
    exprt size = static_cast<const exprt &>(type.size_irep());

    typecheck_expr(size);
    make_constant_index(size);

    mp_integer i;
    if(to_integer(size, i))
    {
      err_location(size);
      throw "failed to convert bit field width";
    }

    // Now converted, set size field.
    type.size(size);

    const typet &base_type=follow(type.subtype());

    if(!base_type.is_signedbv() &&
       !base_type.is_unsignedbv() &&
       !base_type.is_c_enum())
    {
      err_location(type);
      str << "Bit-field with non-integer type: "
          << to_string(base_type);
      throw 0;
    }

    unsigned width=atoi(base_type.width().c_str());

    if(i>width)
    {
      err_location(size);
      throw "bit field size too large";
    }
    else if(i<1)
    {
      err_location(size);
      throw "bit field size too small";
    }

    width=integer2long(i);

    typet tmp(base_type);
    type.swap(tmp);
    type.width(width);
  }
  else if(type.id()=="type_of")
  {
    if(type.is_expression())
    {
      exprt expr = (exprt&) type.subtype();
      typecheck_expr(expr);
      type.swap(expr.type());
    }
    else
    {
      typet t = type.subtype();
      typecheck_type(t);
      type.swap(t);
    }
  }
  else if(type.is_symbol())
  {
    // adjust identifier, if needed
    replace_symbol(type);

    const irep_idt &identifier=type.identifier();

    symbolst::const_iterator s_it=context.symbols.find(identifier);

    if(s_it==context.symbols.end())
    {
      err_location(type);
      str << "type symbol `" << identifier << "' not found";
      throw 0;
    }

    const symbolt &symbol=s_it->second;

    if(!symbol.is_type)
    {
      err_location(type);
      throw "expected type symbol";
    }

    if(symbol.is_macro)
      type=symbol.type; // overwrite
  }
}

/*******************************************************************\

Function: c_typecheck_baset::adjust_function_argument

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::adjust_function_argument(typet &type) const
{
  if(type.is_array() ||
     type.is_incomplete_array())
  {
    type.id("pointer");
    type.remove("size");
    type.remove("#constant");
  }
}
