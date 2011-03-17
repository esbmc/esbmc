/*******************************************************************\

Module: ANSI-C Conversion / Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <iostream>

#include <config.h>
#include <mp_arith.h>
#include <arith_tools.h>
#include <i2string.h>
#include <type_eq.h>
#include <location.h>
#include <string2array.h>
#include <expr_util.h>
#include <prefix.h>
#include <cprover_prefix.h>

#include "c_types.h"
#include "convert-c.h"
#include "c_typecast.h"
#include "unescape_string.h"

/*******************************************************************\

Function: convert_ct::scope2string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string convert_ct::scope2string(const ScopeTbl *scope)
{
  if(scope->level<=FUNCTION_SCOPE)
    return "";

  std::pair<std::map<const ScopeTbl *, unsigned>::const_iterator, bool>
    iresult=scope_map.insert
    (std::pair<const ScopeTbl *, unsigned>(scope, scope_map.size()));

  return i2string(iresult.first->second)+"::";
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(const FunctionDef &functiondef)
{
  // some sanity checks

  if(functiondef.decl==NULL)
  {
    err_location(functiondef);
    throw "function definition has no declatation";
  }

  // convert declaration

  Decl &decl=*functiondef.decl;
  
  symbolt symbol;

  convert(decl, functiondef.location, false, true, symbol);

  if(!decl.form->isFunction())
  {
    err_location(functiondef);
    error("function definition declatation is not a function");
    throw 0;
  }

  // convert type

  FunctionType &function=(FunctionType &)*functiondef.decl->form;  

  if(((FunctionType &)*decl.form).KnR_decl)
  {
    err_location(functiondef);
    error("error: K&R declarations are broken. "
          "Please use ANSI-C style");
    throw 0;
  }
  
  if(decl.is_inline)
    symbol.type.set("#inlined", true);

  // add parameters as local variables

  std::string old_scope=scope_prefix;
  scope_prefix=symbol.name+"::";

  for(int i=0; i<function.nArgs; i++)
  {
    exprt expr;
    convert_Decl(*function.args[i], functiondef.location, expr, true);
  }

  // add names of arguments

  irept &argument_symbols=symbol.type.add("arguments");

  {
    unsigned i=0;
    
    Forall_irep(it, argument_symbols.get_sub())
      if(i<(unsigned)function.nArgs)
      {
        if(function.args[i]->name!=NULL)
        {
          it->set("#identifier", 
            scope_prefix+function.args[i]->name->name);
        }

        i++;
      }
  }

  // convert code to value

  function_name=symbol.name;
  scope_map.clear();
  convert_Statement(functiondef, symbol.value);
  scope_prefix=old_scope;

  symbols.push_back(symbolt());
  symbols.back().swap(symbol);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const Decl &decl,
  const Location &location,
  bool function_argument,
  bool global,
  symbolt &symbol)
{
  switch(decl.storage)
  {
  case ST_Typedef:
    symbol.is_type=true;
    symbol.is_macro=true;
    break;

  case ST_None:
  case ST_Register:
  case ST_Auto:
    break;

  case ST_Static:
    symbol.is_static=true;
    break;

  case ST_Extern:
    symbol.is_extern=true;
    break;

  default:
    err_location(location);
    err << "storage class not supported: ";
    decl.print(err, true);
    throw 0;
  }

  if(decl.form!=NULL)
  {
    std::string old_typedef_name=typedef_name;
    unsigned old_anon_struct_count(anon_struct_count);

    if(symbol.is_type && decl.name!=NULL)
    {
      typedef_name=scope2string(decl.name->entry->scope)+
                   decl.name->name;
      anon_struct_count=0;
    }
 
    convert(*decl.form, location, symbol.type);

    typedef_name=old_typedef_name;
    anon_struct_count=old_anon_struct_count;
  }
  else
  {
    err_location(location);
    err << "error: symbol without type: ";
    if(decl.name!=NULL) err << decl.name->name;
    throw 0;
  }

  if(decl.name==NULL || decl.name->name=="")
    return; // ignore it: may be a struct def

  symbol.base_name=decl.name->name;

  if(symbol.type.id()=="code")
  {
    // add module name for static functions
    if(symbol.is_static)
      symbol.name=language_prefix+module+"::"+symbol.base_name;
    else
      // if it's code, it goes into the global scope,
      // even if it's declared inside a function
      symbol.name=language_prefix+symbol.base_name;
  }
  else if(symbol.is_extern)
  {
    // if it's extern, it goes into the global scope,
    // and it's static, even if it's declared inside a function

    symbol.name="c::"+symbol.base_name;
    symbol.is_static=true;
  }
  else
  {
    symbol.name=scope_prefix;

    // add module name for typedefs
    if(decl.storage==ST_Typedef)
      symbol.name+=module+"::";

    symbol.name+=scope2string(decl.name->entry->scope);
    symbol.name+=symbol.base_name;
  }

  symbol_map.insert(std::pair<SymEntry *, std::string>
    (decl.name->entry, symbol.name));

  // make everyting static if its global and not code
  // and not a type
  if(global && symbol.type.id()!="code" && !symbol.is_type)
    symbol.is_static=true;
    
  convert_location(symbol.location, location);
  
  symbol.is_actual=function_argument;
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(const DeclStemnt &statement)
{
  for(DeclVector::const_iterator
      it=statement.decls.begin();
      it!=statement.decls.end();
      it++)
  {
    if(*it!=NULL)
    {
      symbolt symbol;
      convert(**it, statement.location, false, true, symbol);

      if(symbol.name!="")
      {
        if((**it).initializer!=NULL)
          convert(*(**it).initializer, symbol.value);
          
        symbols.push_back(symbolt());
        symbols.back().swap(symbol);
      }
    }
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(const Block &block)
{
  if(block.isFuncDef())
    convert(static_cast<const FunctionDef &>(block));
  else
  {
    err << "error: unexpected block: " << block;
    throw 0;
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(const Statement &statement)
{
  switch(statement.type)
  {
  case ST_DeclStemnt:
    convert(static_cast<const DeclStemnt &>(statement));
    break;

  case ST_TypedefStemnt:
    err << "error: TypedefStemnt not supported yet: "
        << statement;
    throw 0;

  case ST_Block:
    convert(static_cast<const Block &>(statement));
    break;

  case ST_FileLineStemnt:
  case ST_InclStemnt:
  case ST_EndInclStemnt:
    break;

  default:
    err << "error: statement not expected: "
        << statement;
    throw 0;
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(const TransUnit &unit)
{
  Statement *stemnt;
  for(stemnt=unit.head; stemnt; stemnt=stemnt->next)
    convert(*stemnt);
}

/*******************************************************************\

Function: convert_ct::print_symbolptr_list

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::print_symbolptr_list(
  std::ostream &out,
  const symbolptr_listt &symbolptr_list) const
{
  forall_symbolptr_list(it, symbolptr_list)
    out << "  " << (*it)->name << std::endl;
}

/*******************************************************************\

Function: convert_ct::set_location

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::set_location(irept &dest, const Location &location) const
{
  if(location.line!=0)
    convert_location((locationt &)dest.add("#location"), location);
}

/*******************************************************************\

Function: convert_ct::convert_location

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_location(
  locationt &l,
  const Location &location) const
{
  if(location.file!="")
  {
    std::string unescaped_file;
    unescape_string(location.file, unescaped_file);
    l.set_file(unescaped_file);
  }

  if(location.line!=0)   l.set_line(i2string(location.line));
  if(location.column!=0) l.set_column(i2string(location.column));
  if(function_name!="")  l.set_function(function_name);
}

/*******************************************************************\

Function: convert_c

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool convert_c(
  Project &project,
  std::list<symbolt> &symbols,
  message_handlert &message_handler,
  const std::string &module)
{
  if(project.units.size()==0) return true;

  convert_ct convert_c(symbols, module, message_handler);

  try
  {
    convert_c.convert(*project.units[0]);
  }

  catch(int e)
  {
    convert_c.error();
  }

  catch(const char *e)
  {
    convert_c.error(e);
  }

  catch(const std::string &e)
  {
    convert_c.error(e);
  }

  return convert_c.get_error_found();
}

/*******************************************************************\

Function: convert_c

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool convert_c(
  const Expression &expression,
  exprt &expr,
  message_handlert &message_handler,
  const std::string &module)
{
  std::list<symbolt> symbols;

  convert_ct convert_c(symbols, module, message_handler);

  try
  {
    convert_c.convert(expression, expr);
  }

  catch(int e)
  {
    convert_c.error();
  }

  catch(const char *e)
  {
    convert_c.error(e);
  }

  catch(const std::string &e)
  {
    convert_c.error(e);
  }

  return convert_c.get_error_found();
}

