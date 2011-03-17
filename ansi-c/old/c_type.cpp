/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <hash_cont.h>
#include <lispirep.h>
#include <arith_tools.h>
#include <rename.h>
#include <identifier.h>
#include <location.h>
#include <expr_util.h>
#include <std_types.h>
#include <simplify_expr.h>
#include <i2string.h>

#include "convert-c.h"
#include "c_typecast.h"
#include "config.h"
#include "c_types.h"

/*******************************************************************\

Function: convert_ct::convert_StructType

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_StructType(
  const StructDef &structdef,
  const Location &location,
  typet &dest)
{
  dest.id(structdef.isUnion()?"union":"struct");

  irept &components=dest.add("components");

  hash_set_cont<std::string, string_hash> component_set;

  for(int i=0; i<structdef.nComponents; i++)
  {
    for(const Decl *decl=structdef.components[i];
        decl!=NULL;
        decl=decl->next)
    {
      typet t;

      // convert the type

      if(decl->form==NULL)
      {
        err_location(location);
        throw "struct record without form";
      }

      convert(*decl->form, location, t);

      irept component;
      component.set("type", t);

      // figure out the record name

      if(decl->name==NULL || decl->name->name=="")
      {
        // no record name, that's ok
	components.move_to_sub(component);
      }
      else
      {
        const std::string &name=decl->name->name;

        if(component_set.find(name)!=component_set.end())
        {
          err_location(location);
          throw "error: struct with duplicate record name";
        }

        component_set.insert(name);
        component.set("name", name);
        component.set("pretty_name", name);
        component.set("base_name", name);
	components.move_to_sub(component);
      }
    }
  }
}

/*******************************************************************\

Function: convert_ct::convert_StructType

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_StructType(
  const BaseType &type,
  const Location &location,
  typet &dest)
{
  if(type.tag==NULL)
  {
    // anonymous struct

    if(type.stDefn==NULL)
    {
      err << "error: anonymous struct without StructDef: ";
      type.printBase(err, 0);
      throw 0;
    }

    symbolt symbol;

    convert_StructType(*type.stDefn, location, symbol.type);

    symbol.is_type=true;

    if(scope_prefix==language_prefix)
    {
      if(typedef_name=="")
        symbol.name="anon_struct::#"+module+"::";
      else
        symbol.name="anon_struct::"+typedef_name+"::";
    }
    else
      symbol.name=scope_prefix+"anon_struct::";

    symbol.name+=i2string(anon_struct_count++);

    convert_location(symbol.location, location);
    
    dest.id("symbol");
    dest.set("identifier", symbol.name);
    
    symbols.push_back(symbolt());
    symbols.back().swap(symbol);

    return;
  }

  // with tag, not anonymous

  // first consult symbol map

  symbol_mapt::const_iterator s_it=
    symbol_map.find(type.tag->entry);
    
  bool seen_before;    
  std::string identifier;

  if(s_it!=symbol_map.end())
  {
    identifier=s_it->second;
    seen_before=true;
  }
  else
  {
    identifier=scope_prefix+"struct::"+type.tag->name;
    symbol_map.insert(std::pair<const SymEntry *, std::string>
      (type.tag->entry, identifier));
    seen_before=false;
  }

  dest.id("symbol");
  dest.set("identifier", identifier);

  // does it come with a definition?

  if(type.stDefn!=NULL)
  { 
    // yes!
    
    symbolt symbol;

    symbol.is_type=true;
    symbol.base_name=type.tag->name;
    symbol.name=identifier;
    convert_location(symbol.location, location);
    
    {
      // adjust scope
      std::string old_scope_prefix(scope_prefix);
      scope_prefix+=symbol.base_name+"::";
      unsigned old_anon_struct_count(anon_struct_count);
      anon_struct_count=0;
      
      convert_StructType(*type.stDefn, location, symbol.type);
      symbol.type.set("tag", type.tag->entry->name);
      
      // restore scope
      scope_prefix=old_scope_prefix;
      anon_struct_count=old_anon_struct_count;
    }

    symbols.push_back(symbolt());
    symbols.back().swap(symbol);
  }
  else
  {
    // no definition

    if(!seen_before)
    {
      // not yet there, add it

      symbolt symbol;

      symbol.is_type=true;
      symbol.base_name=type.tag->name;
      symbol.name=identifier;
      convert_location(symbol.location, location);

      symbol.type.id("incomplete_struct");
      symbol.type.set("tag", type.tag->entry->name);

      symbols.push_back(symbolt());
      symbols.back().swap(symbol);
    }
  }
}

/*******************************************************************\

Function: convert_ct::convert_UserType

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert_UserType(
  const BaseType &type,
  const Location &location,
  typet &dest)
{
  if(type.typeName!=NULL)
  {
    if(type.typeName->entry==NULL)
    {
      err_location(location);
      err << "named user type without symbol entry: " 
          << type.typeName->name;
      throw 0;
    }

    if(type.typeName->entry->IsTypeDef())
    {
      convert(type.typeName->entry, location, dest);

      if(type.qualifier&TQ_Volatile)
        dest.set("#volatile", true);
   
      if(type.qualifier&TQ_Const)
        dest.set("#constant", true);

      return;
    }

    err_location(location);
    err << "named user type with unknown symbol entry: " 
        << type.typeName->name;
    throw 0;
  }

  err << "user type not supported: ";
  type.printBase(err, 0);
  throw 0;
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const BaseType &type,
  const Location &location,
  typet &dest)
{
  if(type.qualifier&TQ_Volatile)
    dest.set("#volatile", true);
   
  if(type.qualifier&TQ_Const)
    dest.set("#constant", true);

  if(type.typemask&BT_Bool)
    dest.id("bool");
  else if(type.typemask&BT_Ellipsis)
  {
    dest=typet("ansi_c_ellipsis");
    return;
  }
  else if(type.typemask&BT_Struct ||
          type.typemask&BT_Union)
    return convert_StructType(type, location, dest);
  else if(type.typemask&BT_Enum)
  {
    if(type.typemask!=BT_Enum)
    {
      err << "enum does not take attributes: ";
      type.printBase(err, 0);
      throw 0;
    }

    dest.id("signedbv");
    dest.set("width", config.ansi_c.int_width);
  }
  else if(type.typemask&BT_UserType)
  {
    convert_UserType(type, location, dest);
  }
  else if(type.typemask&BT_Void)
  {
    dest.id("empty");

    if(type.typemask!=BT_Void)
    {
      err << "void does not take attributes: ";
      type.printBase(err, 0);
      throw 0;
    }
  }
  else if(type.typemask&BT_Float)
  {
    dest.id(config.ansi_c.use_fixed_for_float?"fixedbv":"floatbv");
    dest.set("width", config.ansi_c.single_width);

    if(type.typemask!=BT_Float)
    {
      err << "float does not take attributes: ";
      type.printBase(err, 0);
      throw 0;
    }
  }
  else if(type.typemask&BT_Double)
  {
    dest.id(config.ansi_c.use_fixed_for_float?"fixedbv":"floatbv");
    dest.set("width", config.ansi_c.double_width);

    if(type.typemask==BT_Double)
      dest.set("width", config.ansi_c.double_width);
    else if(type.typemask==(BT_Double | BT_Long))
      dest.set("width", config.ansi_c.long_double_width);
    else
    {
      err << "double does not take attributes: ";
      type.printBase(err, 0);
      throw 0;
    }
  }
  else
  {
    if(type.typemask&BT_UnSigned)
      dest.id("unsignedbv");
    else if(type.typemask&BT_Signed)
      dest.id("signedbv");
    else // use defaults
    {
      if(type.typemask&BT_Char)
        dest.id(config.ansi_c.char_is_unsigned?"unsignedbv":"signedbv");
      else
        dest.id("signedbv");
    }

    if(type.typemask&BT_Int8)
      dest.set("width", 8);
    else if(type.typemask&BT_Int16)
      dest.set("width", 16);
    else if(type.typemask&BT_Int32)
      dest.set("width", 32);
    else if(type.typemask&BT_Int64)
      dest.set("width", 64);
    else if(type.typemask&BT_Char)
      dest.set("width", config.ansi_c.char_width);
    else if(type.typemask&BT_Long)
      dest.set("width", config.ansi_c.long_int_width);
    else if(type.typemask&BT_LongLong)
      dest.set("width", config.ansi_c.long_long_int_width);
    else if(type.typemask&BT_Short)
      dest.set("width", config.ansi_c.short_int_width);
    else
      dest.set("width", config.ansi_c.int_width);
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const ArrayType &type, 
  const Location &location,
  typet &dest)
{
  if(type.subType==NULL)
  {
    err << "array without subtype: ";
    type.printBase(err, 0);
    throw 0;
  }
  else
    convert(*type.subType, location, dest.subtype());

  if(dest.subtype().get_bool("#constant"))
    dest.set("#constant", true);

  dest.id("array");

  if(type.size==NULL)
  {
    // array without size: int a[];
    // make it an incomplete type

    dest.id("incomplete_array");
  }
  else
  {
    // a real array

    exprt &size=(exprt &)dest.add("size");
    convert(*type.size, size);
  }
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const BitFieldType &type, 
  const Location &location,
  typet &dest)
{
  if(type.subType==NULL)
  {
    err << "BitFieldType without subtype: ";
    type.printBase(err, 0);
    throw 0;
  }

  typet base_type;
  convert(*type.subType, location, base_type);
  
  if(type.size==NULL)
  {
    err << "BitFieldType without size: ";
    type.printBase(err, 0);
    throw 0;
  }

  exprt size;
  convert(*type.size, size);
  
  dest.id("c_bitfield");
  
  dest.subtype().swap(base_type);
  dest.add("size").swap(size);
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const PtrType &type, 
  const Location &location,
  typet &dest)
{
  dest.id("pointer");

  if(type.subType!=NULL)
    convert(*type.subType, location, dest.subtype());
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const FunctionType &type, 
  const Location &location,
  typet &dest)
{
  dest.id("code");

  irept &arguments=dest.add("arguments");

  for(int i=0; i<type.nArgs; i++)
  {
    code_typet::argumentt argument;

    if(type.args[i]->form==NULL)
      argument.type()=int_type();
    else
      convert(*type.args[i]->form, location, argument.type());

    arguments.move_to_sub(argument);
  }

  const irept::subt &arguments_sub=arguments.get_sub();
  
  unsigned no_arguments=arguments_sub.size();

  if(no_arguments!=0 &&
     arguments_sub.back().find("type").id()==
     "ansi_c_ellipsis")
  {
    // special case: ellipsis
    arguments.get_sub().resize(no_arguments-1);
    arguments.set("ellipsis", true);
  }

  // return type

  if(type.subType!=NULL)
    convert(*type.subType, location, (typet &)dest.add("return_type"));
}

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const Type &type,
  const Location &location,
  typet &dest)
{
  switch(type.type)
  {
   case TT_Base:        // a simple base type, T
    convert((BaseType &)type, location, dest);
    break;

   case TT_Function:
    convert((FunctionType &)type, location, dest);
    break;

   case TT_Pointer:     // pointer to T
    convert((PtrType &)type, location, dest);
    break;

   case TT_Array:       // an array of T
    convert((ArrayType &)type, location, dest);
    break;

   case TT_BitField:    // a bitfield
    convert((BitFieldType &)type, location, dest);
    break;

   default:
    err << "error: unknown type: ";
    type.printBase(err, 0);
    throw 0;
  }

  set_location(dest, location);
}

/*******************************************************************\

Function: convert_ct::do_dynamic_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

#if 0
void convert_ct::do_dynamic_type(
  exprt &dest, 
  const std::string &name,
  typet &type)
{
  if(type.id()=="array")
  {
    do_dynamic_type(dest, name+"_sub", type.subtype());

    exprt &size_expr=(exprt &)type.add("size");
    
    if(!size_expr.is_constant() &&
       size_expr.id()!="infinity")
    {
      const locationt &location=size_expr.location();

      symbolt symbol;

      symbol.type=size_expr.type();
      symbol.type.set("#constant", true);
      symbol.name=name+"_size";
      symbol.lvalue=true;
      symbol.value=gen_zero(symbol.type);
      symbol.location=location;

      identifiert id(symbol.name);
      assert(id.components.size()!=0);
      symbol.base_name=id.components[id.components.size()-1];

      #if 0
      get_new_name(symbol, context);
      #endif

      exprt symbol_expr;
      convert(symbol, location, symbol_expr);

      exprt code("code");
      code.set("statement", "decl");
      code.copy_to_operands(symbol_expr);
      code.move_to_operands(size_expr);
      code.set("#location", location);
      dest.move_to_operands(code);

      size_expr.swap(symbol_expr);

      #if 0
      move_symbol(symbol);
      #endif
    }
  }
  else if(type.id()=="pointer")
  {
  }
  else if(type.id()=="struct")
  {
    const irept::subt &components=type.find("components").get_sub();

    forall_irep(it, components)
    {
      do_dynamic_type(dest, name+"_"+it->get("name"),
                      (typet &)it->find("type"));
    }
  }
}
#endif

/*******************************************************************\

Function: convert_ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void convert_ct::convert(
  const symbolt &symbol,
  const locationt &location,
  typet &dest)
{
  if(symbol.is_macro)
    dest=symbol.type;
  else
  {
    dest.id("symbol");
    dest.set("identifier", symbol.name);
    dest.location()=location;
  }
}
