/*******************************************************************\

Module: ANSI-C Conversion / Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <std_types.h>
#include <prefix.h>

#include "c_typecheck_base.h"
#include "expr2c.h"
#include "type2name.h"
#include "std_types.h"

/*******************************************************************\

Function: c_typecheck_baset::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string c_typecheck_baset::to_string(const exprt &expr)
{
  return expr2c(expr, *this);
}

/*******************************************************************\

Function: c_typecheck_baset::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string c_typecheck_baset::to_string(const typet &type)
{
  return type2c(type, *this);
}

/*******************************************************************\

Function: c_typecheck_baset::replace_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::replace_symbol(irept &symbol)
{
  id_replace_mapt::const_iterator it=
    id_replace_map.find(symbol.identifier());

  if(it!=id_replace_map.end())
    symbol.identifier(it->second);
}

/*******************************************************************\

Function: c_typecheck_baset::move_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_typecheck_baset::move_symbol(symbolt &symbol, symbolt *&new_symbol)
{
  symbol.mode=mode;
  symbol.module=module;
  return context.move(symbol, new_symbol);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_symbol(symbolt &symbol)
{
  // first of all, we do the type
  typecheck_type(symbol.type);

  const typet &final_type=follow(symbol.type);

  // set a few flags
  symbol.lvalue=!symbol.is_type && !symbol.is_macro;

  std::string prefix="c::";
  std::string root_name=prefix+id2string(symbol.base_name);
  std::string new_name=id2string(symbol.name);

  // do anon-tags first
  if(symbol.is_type &&
     has_prefix(id2string(symbol.name), prefix+"tag-#anon"))
  {
    // used to be file local:
    // new_name=prefix+module+"::tag-"+id2string(symbol.base_name);

    // now: rename them
    std::string typestr = type2name(symbol.type);
    new_name = prefix+"tag-#anon#" + typestr;
    id_replace_map[symbol.name]=new_name;

    symbolt* s = context.find_symbol(new_name);
    if(s != nullptr)
      return; // bail out, we have an appropriate symbol already.

    irep_idt newtag((std::string("#anon#") + typestr).c_str());
    symbol.type.tag(newtag);
  }
  else if(symbol.file_local) // rename file-local stuff
  {
    // strip prefix
    assert(has_prefix(id2string(symbol.name), prefix));

    new_name=prefix+module+"::"+
      std::string(id2string(symbol.name), prefix.size(), std::string::npos);
  }
  else if(symbol.is_extern && !final_type.is_code())
  {
    // variables mared as "extern" go into the global namespace
    // and have static lifetime
    new_name=root_name;
    symbol.static_lifetime=true;
  }
  else if(final_type.is_code())
  {
    // functions always go into the global namespace
    // code doesn't have lifetime
    new_name=root_name;
    symbol.static_lifetime=false;
  }

  if(symbol.name!=new_name)
  {
    id_replace_map[symbol.name]=new_name;
    symbol.name=new_name;
  }

  // set the pretty name
  if(symbol.is_type &&
     (final_type.id()=="struct" ||
      final_type.id()=="incomplete_struct"))
  {
    symbol.pretty_name="struct "+id2string(symbol.base_name);
  }
  else if(symbol.is_type &&
          (final_type.id()=="union" ||
           final_type.id()=="incomplete_union"))
  {
    symbol.pretty_name="union "+id2string(symbol.base_name);
  }
  else if(symbol.is_type &&
          (final_type.id()=="c_enum" ||
           final_type.id()=="incomplete_c_enum"))
  {
    symbol.pretty_name="enum "+id2string(symbol.base_name);
  }
  else
  {
    // just strip the c::
    symbol.pretty_name=
      std::string(new_name, prefix.size(), std::string::npos);
  }

  // see if we have it already
  symbolt *s = context.find_symbol(symbol.name);
  if(s == nullptr)
  {
    // just put into context
    symbolt *new_symbol;
    bool res = move_symbol(symbol, new_symbol);
    assert(!res);
    (void)res; //ndebug

    typecheck_new_symbol(*new_symbol);
  }
  else
    typecheck_symbol_redefinition(*s, symbol);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_new_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_new_symbol(symbolt &symbol)
{
  if(symbol.is_parameter)
    adjust_function_argument(symbol.type);

  // check initializer, if needed

  if(symbol.type.is_code())
  {
    if(symbol.value.is_not_nil())
      typecheck_function_body(symbol);
    else
    {
      // we don't need the identifiers
      code_typet &code_type=to_code_type(symbol.type);
      for(code_typet::argumentst::iterator
          it=code_type.arguments().begin();
          it!=code_type.arguments().end();
          it++)
        it->set_identifier("");
    }
  }
  else if(symbol.type.id()=="incomplete_array" ||
          symbol.type.is_array())
  {
    // insert a new type symbol for the array
    {
      symbolt new_symbol;
      new_symbol.name=id2string(symbol.name)+"$type";
      new_symbol.base_name=id2string(symbol.base_name)+"$type";
      new_symbol.location=symbol.location;
      new_symbol.mode=symbol.mode;
      new_symbol.module=symbol.module;
      new_symbol.type=symbol.type;
      new_symbol.is_type=true;
      new_symbol.is_macro=true;

      symbol.type=symbol_typet(new_symbol.name);

      symbolt *new_sp;
      context.move(new_symbol, new_sp);
    }

    do_initializer(symbol);
  }
  else
  {
    // check the initializer
    do_initializer(symbol);
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_symbol_redefinition

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_symbol_redefinition(
  symbolt &old_symbol,
  symbolt &new_symbol)
{
  const typet &final_old=follow(old_symbol.type);
  const typet &final_new=follow(new_symbol.type);

  // K&R stuff?
  if(old_symbol.type.id()=="KnR")
  {
    // check the type
    if(final_new.is_code())
    {
      err_location(new_symbol.location);
      throw "function type not allowed for K&R function argument";
    }

    // fix up old symbol -- we now got the type
    old_symbol.type=new_symbol.type;
  }
  else if(old_symbol.is_type)
  {
    // see if we had s.th. incomplete before
    if(old_symbol.type.id()=="incomplete_struct" ||
       old_symbol.type.id()=="incomplete_union" ||
       old_symbol.type.id()=="incomplete_c_enum")
    {
      // new one complete?
      if("incomplete_"+new_symbol.type.id_string()==old_symbol.type.id_string())
      {
        // overwrite location
        old_symbol.location=new_symbol.location;

        // move body
        old_symbol.type.swap(new_symbol.type);
      }
      else if(new_symbol.type.id()==old_symbol.type.id())
        return;
      else
      {
        err_location(new_symbol.location);
        str << "error: conflicting defintion of type symbol `"
            << new_symbol.display_name()
            << "'";
        throw 0;
      }
    }
    else
    {
      // see if new one is just a tag
      if(new_symbol.type.id()=="incomplete_struct" ||
         new_symbol.type.id()=="incomplete_union" ||
         new_symbol.type.id()=="incomplete_c_enum")
      {
        if("incomplete_"+old_symbol.type.id_string()==new_symbol.type.id_string())
        {
          // just ignore silently
        }
        else
        {
          // arg! new tag type
          err_location(new_symbol.location);
          str << "error: conflicting defintion of tag symbol `"
              << new_symbol.display_name()
              << "'";
          throw 0;
        }
      }
      else
      {
        // see if it changed
        if(new_symbol.type!=old_symbol.type)
        {
          err_location(new_symbol.location);
          str << "error: type symbol `" << new_symbol.display_name()
              << "' defined twice:" << std::endl;
          str << "Original: " << to_string(old_symbol.type) << std::endl;
          str << "     New: " << to_string(new_symbol.type);
          throw 0;
        }
      }
    }
  }
  else
  {
    bool inlined=new_symbol.type.is_code() &&
         (new_symbol.type.inlined() ||
          old_symbol.type.inlined());

    if(final_old!=final_new)
    {
      if(final_old.is_array() &&
         final_new.id()=="incomplete_array" &&
         final_old.subtype()==final_new.subtype())
      {
        // this is ok
        new_symbol.type=old_symbol.type;
      }
      else if(final_old.id()=="incomplete_array" &&
              final_new.is_array() &&
              final_old.subtype()==final_new.subtype())
      {
        // this is also ok
        if (old_symbol.type.id()=="symbol")
        {
          // fix the symbol, not just the type
          const irep_idt ident = old_symbol.type.identifier();
          symbolt* s = context.find_symbol(ident);
          if(s == nullptr)
          {
            err_location(old_symbol.location);
            str << "failed to find symbol `" << ident << "'";
            throw 0;
          }

          symbolt &symbol = *s;
          symbol.type=final_new;
        }
        else
          old_symbol.type=new_symbol.type;
      }
      else if(old_symbol.type.is_code() &&
              new_symbol.type.is_code())
      {
        code_typet &old_ct=to_code_type(old_symbol.type);
        code_typet &new_ct=to_code_type(new_symbol.type);

        if(old_ct.has_ellipsis() && !new_ct.has_ellipsis())
          old_ct=new_ct;
        else if(!old_ct.has_ellipsis() && new_ct.has_ellipsis())
          new_ct=old_ct;
      }
      else if((final_old.id()=="incomplete_c_enum" ||
               final_old.id()=="c_enum") &&
              (final_new.id()=="incomplete_c_enum" ||
               final_new.id()=="c_enum"))
      {
        // this is ok for now
      }
      else
      {
        err_location(new_symbol.location);
        str << "error: symbol `" << new_symbol.display_name()
            << "' defined twice with different types:" << std::endl;
        str << "Original: " << to_string(old_symbol.type) << std::endl;
        str << "     New: " << to_string(new_symbol.type);
        throw 0;
      }
    }
    else // finals are equal
      if(old_symbol.type.id()=="symbol" &&
         new_symbol.type.id()=="incomplete_array")
        new_symbol.type=old_symbol.type; // fix from i.a. to a symbol ref.

    if(inlined)
    {
      old_symbol.type.inlined(true);
      new_symbol.type.inlined(true);
    }

    // do value
    if(old_symbol.type.is_code())
    {
      if(new_symbol.value.is_not_nil())
      {
        if(old_symbol.value.is_not_nil())
        {
          err_location(new_symbol.location);
          str << "function `" << new_symbol.display_name()
              << "' defined twice";
          error();
        }
        else
        {
          typecheck_function_body(new_symbol);

          // overwrite location
          old_symbol.location=new_symbol.location;

          // move body
          old_symbol.value.swap(new_symbol.value);

          // overwrite type (because of argument names)
          old_symbol.type=new_symbol.type;
        }
      }
    }
    else
    {
      // initializer
      do_initializer(new_symbol);

      if(new_symbol.value.is_not_nil())
      {
        // see if we already have one
        if(old_symbol.value.is_not_nil())
        {
          if(new_symbol.value.zero_initializer())
          {
            // do nothing
          }
          else if(old_symbol.value.zero_initializer())
          {
            old_symbol.value=new_symbol.value;
            old_symbol.type=new_symbol.type;
          }
          else
          {
            if(new_symbol.is_macro &&
               (final_new.id()=="incomplete_c_enum" ||
                final_new.id()=="c_enum") &&
                old_symbol.value.is_constant() &&
                new_symbol.value.is_constant() &&
                old_symbol.value.value()==new_symbol.value.value())
            {
              // ignore
            }
            else
            {
              err_location(new_symbol.value);
              str << "symbol `" << new_symbol.display_name()
                  << "' already has an initial value";
              warning();
            }
          }
        }
        else
        {
          old_symbol.value=new_symbol.value;
          old_symbol.type=new_symbol.type;
        }
      }
    }
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_function_body

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_function_body(symbolt &symbol)
{
  code_typet &code_type=to_code_type(symbol.type);

  // adjust the function identifiers
  for(code_typet::argumentst::iterator
      a_it=code_type.arguments().begin();
      a_it!=code_type.arguments().end();
      a_it++)
  {
    irep_idt identifier=a_it->get_identifier();
    if(identifier!="")
    {
      id_replace_mapt::const_iterator
        m_it=id_replace_map.find(identifier);

      if(m_it!=id_replace_map.end())
        a_it->set_identifier(m_it->second);
    }
  }

  assert(symbol.value.is_not_nil());

  // fix type
  symbol.value.type()=code_type;

  // set return type
  return_type=code_type.return_type();

  typecheck_code(to_code(symbol.value));

  if(symbol.name=="c::main")
    add_argc_argv(symbol);
}
