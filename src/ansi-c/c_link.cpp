/*******************************************************************\

Module: ANSI-C Linking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <location.h>
#include <namespace.h>
#include <typecheck.h>
#include <base_type.h>
#include <i2string.h>

#include "expr2c.h"
#include "c_link.h"
#include "fix_symbol.h"

class c_linkt:public typecheckt
{
public:
  c_linkt(
    contextt &_context,
    contextt &_new_context,
    const std::string &_module,
    message_handlert &_message_handler):
    typecheckt(_message_handler),
    context(_context),
    new_context(_new_context),
    module(_module),
    ns(_context, _new_context),
    type_counter(0)
  {
    context.Foreach_operand(
      [this] (symbolt& s)
      {
        if (!s.module.empty())
          known_modules.insert(s.module);
      }
    );
  }

  virtual void typecheck();

protected:
  void duplicate(symbolt &in_context, symbolt &new_symbol);
  void duplicate_type(symbolt &in_context, symbolt &new_symbol);
  void duplicate_symbol(symbolt &in_context, symbolt &new_symbol);
  void move(symbolt &new_symbol);

  // overload to use language specific syntax
  virtual std::string to_string(const exprt &expr);
  virtual std::string to_string(const typet &type);

  contextt &context;
  contextt &new_context;
  std::string module;
  namespacet ns;

  typedef hash_set_cont<irep_idt, irep_id_hash> known_modulest;
  known_modulest known_modules;

  fix_symbolt symbol_fixer;

  unsigned type_counter;
};

/*******************************************************************\

Function: c_linkt::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string c_linkt::to_string(const exprt &expr)
{
  return expr2c(expr, ns);
}

/*******************************************************************\

Function: c_linkt::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string c_linkt::to_string(const typet &type)
{
  return type2c(type, ns);
}

/*******************************************************************\

Function: c_linkt::duplicate

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_linkt::duplicate(
  symbolt &in_context,
  symbolt &new_symbol)
{
  if(new_symbol.is_type!=in_context.is_type)
  {
    str << "class conflict on symbol `" << in_context.name
        << "'";
    throw 0;
  }

  if(new_symbol.is_type)
    duplicate_type(in_context, new_symbol);
  else
    duplicate_symbol(in_context, new_symbol);
}

/*******************************************************************\

Function: c_linkt::duplicate_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_linkt::duplicate_type(
  symbolt &in_context,
  symbolt &new_symbol)
{
  // check if it is the same -- use base_type_eq
  if(!base_type_eq(in_context.type, new_symbol.type, ns))
  {
    if(in_context.type.id()=="incomplete_struct" &&
       new_symbol.type.id()=="struct")
    {
      // replace old symbol
      in_context.type=new_symbol.type;
    }
    else if(in_context.type.id()=="struct" &&
            new_symbol.type.id()=="incomplete_struct")
    {
      // ignore
    }
    else if(in_context.type.id()=="struct" &&
            new_symbol.type.id()=="incomplete_struct")
    {
      // ignore
    }
    else if(ns.follow(in_context.type).id()=="incomplete_array" &&
            ns.follow(new_symbol.type).is_array())
    {
      // store new type
      in_context.type=new_symbol.type;
    }
    else if(ns.follow(in_context.type).is_array() &&
            ns.follow(new_symbol.type).id()=="incomplete_array")
    {
      // ignore
    }
    else
    {
      // rename, there are no type clashes in C
      irep_idt old_identifier=new_symbol.name;

      do
      {
        irep_idt new_identifier=
          id2string(old_identifier)+"#link"+i2string(type_counter++);

        new_symbol.name=new_identifier;
      }
      while(context.move(new_symbol));
    }
  }
}

/*******************************************************************\

Function: c_linkt::duplicate_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_linkt::duplicate_symbol(
  symbolt &in_context,
  symbolt &new_symbol)
{
  // see if it is a function or a variable

  bool is_code_in_context=in_context.type.is_code();
  bool is_code_new_symbol=new_symbol.type.is_code();

  if(is_code_in_context!=is_code_new_symbol)
  {
    err_location(new_symbol.location);
    str << "error: conflicting definition for symbol \""
        << in_context.display_name()
        << "\"" << std::endl;
    str << "old definition: " << to_string(in_context.type)
        << std::endl;
    str << "Module: " << in_context.module << std::endl;
    str << "new definition: " << to_string(new_symbol.type)
        << std::endl;
    str << "Module: " << new_symbol.module;
    throw 0;
  }

  if(is_code_in_context)
  {
    // both are functions

    // we don't compare the types, they will be too different

    // care about code

    if(!new_symbol.value.is_nil())
    {
      if(in_context.value.is_nil())
      {
        // the one with body wins!
        in_context.value.swap(new_symbol.value);
        in_context.type.swap(new_symbol.type); // for argument identifiers
      }
      else if(in_context.type.inlined())
      {
        // ok
      }
      else if(base_type_eq(in_context.type, new_symbol.type, ns))
      {
        // keep the one in in_context -- libraries come last!
        str << "warning: function `" << in_context.name << "' in module `" <<
          new_symbol.module << "' is shadowed by a definition in module `" <<
          in_context.module << "'";
        warning();
      }
      else
      {
        err_location(new_symbol.value);
        str << "error: duplicate definition of function `"
            << in_context.name
            << "'" << std::endl;
        str << "In module `" << in_context.module
            << "' and module `" << new_symbol.module << "'";
        throw 0;
      }
    }
  }
  else
  {
    // both are variables

    if(!base_type_eq(in_context.type, new_symbol.type, ns))
    {
      const typet &old_type = ns.follow(in_context.type);
      const typet &new_type = ns.follow(new_symbol.type);

      if(old_type.is_incomplete_array() &&
         new_type.is_array())
      {
        // store new type
        in_context.type=new_symbol.type;
      }
      else if(old_type.is_pointer() && new_type.is_array())
      {
        // store new type
        in_context.type=new_symbol.type;
      }
      else if(old_type.is_array() && new_type.is_pointer())
      {
        // ignore
      }
      else if(old_type.is_array() &&
              new_type.is_incomplete_array())
      {
        // ignore
      }
      else if(old_type.id()=="incomplete_struct" &&
              new_type.is_struct())
      {
        // store new type
        in_context.type=new_symbol.type;
      }
      else if(old_type.is_struct() &&
              new_type.id()=="incomplete_struct")
      {
        // ignore
      }
      else if(old_type.is_pointer() &&
              new_type.is_incomplete_array())
      {
        // ignore
      }
      else
      {
        err_location(new_symbol.location);
        str << "error: conflicting definition for variable `"
            << in_context.name
            << "'" << std::endl;
        str << "old definition: " << to_string(in_context.type)
            << std::endl;
        str << "Module: " << in_context.module << std::endl;
        str << "new definition: " << to_string(new_symbol.type)
            << std::endl;
        str << "Module: " << new_symbol.module;
        throw 0;
      }
    }

    // care about initializers

    if(!new_symbol.value.is_nil() &&
       !new_symbol.value.zero_initializer())
    {
      if(in_context.value.is_nil() ||
         in_context.value.zero_initializer())
      {
        in_context.value.swap(new_symbol.value);
      }
      else if(!base_type_eq(in_context.value, new_symbol.value, ns))
      {
        err_location(new_symbol.value);
        str << "error: conflicting initializers for variable `"
            << in_context.name
            << "'" << std::endl;
        str << "old value: " << to_string(in_context.value)
            << std::endl;
        str << "Module: " << in_context.module << std::endl;
        str << "new value: " << to_string(new_symbol.value)
            << std::endl;
        str << "Module: " << new_symbol.module;
        throw 0;
      }
    }
  }
}

/*******************************************************************\

Function: c_linkt::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_linkt::typecheck()
{
  new_context.Foreach_operand(
    [this] (symbolt& s)
    {
      // build module clash table
      if(s.file_local && known_modules.find(s.module) != known_modules.end())
      {
        // we could have a clash
        unsigned counter=0;
        std::string newname = id2string(s.name);

        while(context.find_symbol(newname) != nullptr)
        {
          // there is a clash, rename!
          counter++;
          newname = id2string(s.name) + "#-mc-" + i2string(counter);
        }

        if(counter > 0)
        {
          exprt subst("symbol");
          subst.identifier(newname);
          subst.location() = s.location;
          symbol_fixer.insert(
            s.name,
            static_cast<const typet&>(static_cast<const irept&>(subst)));
          subst.type() = s.type;
          symbol_fixer.insert(s.name, subst);
        }
      }
    }
  );

  symbol_fixer.fix_context(new_context);

  new_context.Foreach_operand_in_order(
    [this] (symbolt& s)
    {
      symbol_fixer.fix_symbol(s);
      move(s);
    }
  );
}

/*******************************************************************\

Function: c_linkt::move

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_linkt::move(symbolt &new_symbol)
{
  // try to add it

  symbolt *new_symbol_ptr;
  if(context.move(new_symbol, new_symbol_ptr))
    duplicate(*new_symbol_ptr, new_symbol);
}

/*******************************************************************\

Function: convert_c

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_link(
  contextt &context,
  contextt &new_context,
  message_handlert &message_handler,
  const std::string &module)
{
  c_linkt c_link(context, new_context, module, message_handler);
  return c_link.typecheck_main();
}
