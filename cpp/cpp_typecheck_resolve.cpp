/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <expr_util.h>
#include <std_types.h>
#include <std_expr.h>
#include <i2string.h>
#include <arith_tools.h>
#include <prefix.h>

#include "cpp_typecheck.h"
#include "cpp_typecheck_resolve.h"
#include "cpp_template_type.h"
#include "cpp_type2name.h"
#include "cpp_util.h"

/*******************************************************************\

Function: cpp_typecheck_resolvet::cpp_typecheck_resolvet

Inputs:

Outputs:

Purpose:

\*******************************************************************/

cpp_typecheck_resolvet::cpp_typecheck_resolvet(cpp_typecheckt &_cpp_typecheck):
  cpp_typecheck(_cpp_typecheck),
  this_expr(_cpp_typecheck.cpp_scopes.current_scope().this_expr)
{
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::convert_identifiers

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::convert_identifiers(
  const cpp_scopest::id_sett &id_set,
  const locationt &location,
  const irept &template_args,
  const cpp_typecheck_fargst &fargs,
  resolve_identifierst &identifiers)
{
  for(cpp_scopest::id_sett::const_iterator
      it=id_set.begin();
      it!=id_set.end();
      it++)
  {
    exprt e;
    const cpp_idt &identifier=**it;

    convert_identifier(identifier, location, template_args, fargs, e);

    if(e.is_not_nil())
    {
      identifiers.insert(e);
    }
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::apply_template_args

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::apply_template_args(
  resolve_identifierst &identifiers,
  const irept &template_args,
  const cpp_typecheck_fargst &fargs)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator
      it=old_identifiers.begin();
      it!=old_identifiers.end();
      it++)
  {
    exprt e=*it;
    apply_template_args(e, template_args, fargs);

    if(e.is_not_nil())
    {
      if(e.id()=="type")
        assert(e.type().is_not_nil());

      identifiers.insert(e);
    }
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::remove_templates

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::remove_templates(
  resolve_identifierst &identifiers)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator
      it=old_identifiers.begin();
      it!=old_identifiers.end();
      it++)
  {
    if(!cpp_typecheck.follow(it->type()).get_bool("is_template"))
      identifiers.insert(*it);
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::remove_duplicates

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::remove_duplicates(
  resolve_identifierst &identifiers)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  std::set<irep_idt> ids;
  std::set<exprt> other;

  for(resolve_identifierst::const_iterator
      it=old_identifiers.begin();
      it!=old_identifiers.end();
      it++)
  {
    irep_idt id;

    if(it->id()=="symbol")
      id=it->get("identifier");
    else if(it->id()=="type" && it->type().id()=="symbol")
      id=it->type().get("identifier");

    if(id=="")
    {
      if(other.insert(*it).second)
        identifiers.insert(*it);
    }
    else
    {
      if(ids.insert(id).second)
        identifiers.insert(*it);
    }
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::convert_template_argument

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::convert_template_argument(
                                                       const cpp_idt &identifier,
                                                       const locationt &location,
                                                       const irept &template_args,
                                                       exprt &e)
{
  // template arguments never have template arguments themselves
  if(template_args.is_not_nil())
    return;

  // look up in template map
  e=cpp_typecheck.template_map.lookup(identifier.identifier);

  if(e.is_nil())
  {
    #if 1
    cpp_typecheck.err_location(location);
    cpp_typecheck.str<< "internal error: template argument not found: " << identifier << std::endl;
    throw 0;
    #endif

    // make it a symbol
    const symbolt &symbol=
    cpp_typecheck.lookup(identifier.identifier);

    if(symbol.is_type)
    {
      e=type_exprt(symbol_typet(symbol.name));
    }
    else
      e=cpp_symbol_expr(symbol);
  }

  e.location()=location;
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::convert_identifier

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::convert_identifier(
  const cpp_idt &identifier,
  const locationt &location,
  const irept &template_args,
  const cpp_typecheck_fargst &fargs,
  exprt &e)
{
  e.make_nil();

  if(identifier.id_class==cpp_scopet::TEMPLATE_ARGUMENT)
  {
    convert_template_argument(identifier, location, template_args, e);
    return;
  }

  if(identifier.is_member &&
     !identifier.is_constructor &&
     !identifier.is_static_member)
  {
    const symbolt &class_symbol=
      cpp_typecheck.lookup(identifier.class_identifier);

    const struct_typet &struct_type = to_struct_type(class_symbol.type);

    const exprt component=struct_type.get_component(identifier.identifier);
    const typet &type=component.type();

    if(identifier.id_class==cpp_scopet::TYPEDEF)
    {
      e=type_exprt(type);
    }
    else if(identifier.id_class==cpp_scopet::SYMBOL)
    {
      e=exprt("member");
      e.set("component_name", identifier.identifier);
      e.location() = location;

      exprt object;
      object.make_nil();

      #if 0
      std::cout << "I: " << identifier.class_identifier
      << " " << cpp_typecheck.cpp_scopes.current_scope().this_class_identifier << std::endl;
      #endif

      // find the object of the member expression
      if(class_symbol.type.find("#unnamed_object").is_not_nil())
      {
        cpp_scopet::id_sett id_set;
        cpp_typecheck.cpp_scopes.current_scope().recursive_lookup(
                                                                  class_symbol.type.get("#unnamed_object"),
                                                                  id_set);

        assert(id_set.size()==1);

        irept nil;
        nil.make_nil();
        convert_identifier(**(id_set.begin()),
                           location,nil, fargs, object);
        assert(object.is_not_nil());
      }
      else if(fargs.has_object)
      {
        object = fargs.operands.front();
      }
      else
      {
        if(component.get_bool("is_operator") &&
           fargs.operands.size()==to_code_type(component.type()).arguments().size())
        {
          // turn  'OP(a, b)' into 'a.opratorOP(b)'
          object = fargs.operands.front();
        }
        else if(this_expr.is_not_nil())
        {
          // use this->...
          assert(this_expr.type().id()=="pointer");
          object=exprt("dereference", this_expr.type().subtype());
          object.copy_to_operands(this_expr);
          object.type().set("#constant",
                            this_expr.type().subtype().get_bool("#constant"));
          object.set("#lvalue",true);
          object.location()=location;
        }
      }

      // check if the object has the right member
      typet object_type = cpp_typecheck.follow(object.type());
      if(object_type.id() == "struct" || object_type.id() == "union" )
      {
        const struct_typet& object_struct = to_struct_type(object_type);
        if(object_struct.get_component(identifier.identifier.c_str()).is_nil())
          object.make_nil();
      }
      else object.make_nil();

      if(object.is_not_nil())
      {
        // we got an object
        e.move_to_operands(object);

        bool old_value = cpp_typecheck.disable_access_control;
        cpp_typecheck.disable_access_control = true;
        cpp_typecheck.typecheck_expr_member(e);
        cpp_typecheck.disable_access_control = old_value;
      }
      else
      {
        // this has to be a method
        if(identifier.is_method && !fargs.in_use)
          e = cpp_symbol_expr(cpp_typecheck.lookup(identifier.identifier));
        else
        {
          e.make_nil();
        }
      }
    }
  }
  else
  {
    const symbolt &symbol=
      cpp_typecheck.lookup(identifier.identifier);

    if(symbol.is_type)
    {
      e=type_exprt();

      if(symbol.is_macro)
      {
        e.type()=symbol.type;
      }
      else
      {
        e.type()=typet("symbol");
        e.type().set("identifier", symbol.name);
      }
    }
    else
    {
      if(symbol.is_macro)
      {
        e=symbol.value;
        assert(e.is_not_nil());
      }
      else
      {

        typet followed = symbol.type;
        bool constant = followed.get_bool("#constant");

        while(followed.id() == "symbol")
        {
          typet tmp = cpp_typecheck.lookup(followed).type;
          followed = tmp;
          constant |= followed.get_bool("#constant");
        }

        if(constant &&
           symbol.value.is_not_nil() &&
         is_number(followed) &&
         symbol.value.id() == "constant")
        {
          e=symbol.value;
        }
        else
        {
          e=cpp_symbol_expr(symbol);
        }
      }
    }
  }

  e.location()=location;
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::disambiguate

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::filter(
  resolve_identifierst &identifiers,
  const wantt want)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  for(resolve_identifierst::const_iterator
      it=old_identifiers.begin();
      it!=old_identifiers.end();
      it++)
  {
    bool match=false;

    switch(want)
    {
    case TYPE:
      match=(it->id()=="type");
      break;

    case VAR:
      match=(it->id()!="type");
      break;

    case BOTH:
      match=true;
      break;

    default:
      assert(false);
      break;
    }

    if(match)
      identifiers.insert(*it);
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::disambiguate

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::disambiguate(
  resolve_identifierst &identifiers,
  const cpp_typecheck_fargst &fargs)
{
  resolve_identifierst old_identifiers;
  old_identifiers.swap(identifiers);

  // sort according to distance
  std::multimap<unsigned, exprt> distance_map;

  for(resolve_identifierst::const_iterator
      it=old_identifiers.begin();
      it!=old_identifiers.end();
      it++)
  {
    unsigned args_distance;

    if(disambiguate(*it, args_distance, fargs))
    {
      unsigned template_distance=0;

      if(it->type().get("#template")!="")
        template_distance=it->type().
          find("#template_arguments").find("arguments").get_sub().size();

      // we give strong preference to functions that have
      // fewer template arguments
      unsigned total_distance=
        1000*template_distance+args_distance;

      distance_map.insert(
        std::pair<unsigned, exprt>(total_distance, *it));
    }
  }

  identifiers.clear();

  // put in top ones
  if(!distance_map.empty())
  {
    unsigned distance=distance_map.begin()->first;

    for(std::multimap<unsigned, exprt>::const_iterator
        it=distance_map.begin();
        it!=distance_map.end() && it->first==distance;
        it++)
    identifiers.insert(it->second);
  }

  if(identifiers.size() > 1 && fargs.in_use)
  {
    // try to further disambiguate functions

    for(resolve_identifierst::iterator
        it1 = identifiers.begin();
        it1 != identifiers.end();
        it1++)
    {
        if(it1->type().id()!="code") continue;

        const code_typet& f1 =
          to_code_type(it1->type());

        for(resolve_identifierst::iterator it2=
            identifiers.begin();
            it2 != identifiers.end();
            ) // no it2++
        {
          if(it1 == it2)
          {
            it2++;
            continue;
          }

          if(it2->type().id()!="code")
          {
            it2++;
            continue;
          }

          const code_typet& f2 =
            to_code_type(it2->type());

          // TODO: may fail when using ellipsis
          assert(f1.arguments().size() == f2.arguments().size());

          bool f1_better = true;
          bool f2_better = true;

          for(unsigned i=0; i < f1.arguments().size() && (f1_better || f2_better); i++)
          {
            typet type1 = f1.arguments()[i].type();
            typet type2 = f2.arguments()[i].type();

            if(type1 == type2)
              continue;

            if(is_reference(type1) != is_reference(type2))
              continue;

            if(type1.id() == "pointer")
            {
              typet tmp = type1.subtype();
              type1 = tmp;
            }

            if(type2.id() == "pointer")
            {
              typet tmp = type2.subtype();
              type2 = tmp;
            }

            const typet& followed1 = cpp_typecheck.follow(type1);
            const typet& followed2 = cpp_typecheck.follow(type2);

            if(followed1.id() != "struct" || followed2.id() != "struct")
              continue;

            const struct_typet& struct1 = to_struct_type(followed1);
            const struct_typet& struct2 = to_struct_type(followed2);

            if(f1_better && cpp_typecheck.subtype_typecast(struct1, struct2))
            {
              f2_better = false;
            }
            else if(f2_better && cpp_typecheck.subtype_typecast(struct2,struct1))
            {
              f1_better = false;
            }
          }

          resolve_identifierst::iterator prev_it = it2;
          it2++;

          if(f1_better && !f2_better)
            identifiers.erase(prev_it);
        }
    }
  }

}

/*******************************************************************\

Function: cpp_typecheck_resolvet::make_constructors

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::make_constructors(
                                               resolve_identifierst &identifiers)
{
  resolve_identifierst new_identifiers;

  resolve_identifierst::iterator next;

  for(resolve_identifierst::iterator
      it=identifiers.begin();
      it!=identifiers.end();
      it++)
  {
    if(it->id()=="type")
    {
      const typet& symbol_type =
        cpp_typecheck.follow(it->type());

      if(symbol_type.id() != "struct")
      {
        // it's ok
        new_identifiers.insert(*it);
        continue;
      }

      if(cpp_typecheck.cpp_is_pod(symbol_type))
      {
        // in that case, there is no constructor to call
        new_identifiers.insert(*it);
        continue;
      }

      struct_typet struct_type = to_struct_type(symbol_type);

      const struct_typet::componentst &components =
        struct_type.components();

      // go over components
      for(struct_typet::componentst::const_iterator
          itc=components.begin();
          itc!=components.end();
          itc++)
      {
        const struct_typet::componentt &component = *itc;
        const typet &type=component.type();

        if(component.get_bool("from_base"))
          continue;

        if(type.find("return_type").id()=="constructor")
        {
          const symbolt &symb =
            cpp_typecheck.lookup(component.get_name());
          exprt e = cpp_symbol_expr(symb);
          e.type() = type;
          new_identifiers.insert(e);
        }
      }
    }
    else
      new_identifiers.insert(*it);
  }

  identifiers = new_identifiers;
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::do_builtin

Inputs:

Outputs:

Purpose:

\*******************************************************************/

exprt cpp_typecheck_resolvet::do_builtin(
  const irep_idt &base_name,
  irept &template_args)
{
  exprt dest;

  irept::subt &arguments=
    template_args.add("arguments").get_sub();

  if(base_name=="unsignedbv" ||
     base_name=="signedbv")
  {
    if(arguments.size()!=1)
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name)+" expects one template argument, "
        "but got "+i2string((unsigned long)arguments.size());
    }

    exprt &argument=static_cast<exprt &>(arguments.front());

    if(argument.id()=="type")
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name)+" expects one integer template argument, "
      "but got type";
    }

    cpp_typecheck.typecheck_expr(argument);

    mp_integer i;
    if(to_integer(argument, i))
    {
      cpp_typecheck.err_location(location);
      throw "template argument must be constant";
    }

    if(i<1)
    {
      cpp_typecheck.err_location(location);
      throw "template argument must be greater than zero";
    }

    dest=type_exprt(typet(base_name));
    dest.type().set("width", integer2string(i));
  }
  else if(base_name=="fixedbv")
  {
    if(arguments.size()!=2)
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name)+" expects two template arguments, "
        "but got "+i2string((unsigned long)arguments.size());
    }

    exprt &argument0=static_cast<exprt &>(arguments.front());
    exprt &argument1=static_cast<exprt &>(*(++arguments.begin()));

    if(argument0.id()=="type")
    {
      cpp_typecheck.err_location(argument0);
      throw id2string(base_name)+" expects two integer template arguments, "
        "but got type";
    }

    if(argument1.id()=="type")
    {
      cpp_typecheck.err_location(argument1);
      throw id2string(base_name)+" expects two integer template arguments, "
        "but got type";
    }

    cpp_typecheck.typecheck_expr(argument0);
    cpp_typecheck.typecheck_expr(argument1);

    mp_integer width, integer_bits;

    if(to_integer(argument0, width))
    {
      cpp_typecheck.err_location(argument0);
      throw "template argument must be constant";
    }

    if(to_integer(argument1, integer_bits))
    {
      cpp_typecheck.err_location(argument1);
      throw "template argument must be constant";
    }

    if(width<1)
    {
      cpp_typecheck.err_location(argument0);
      throw "template argument must be greater than zero";
    }

    if(integer_bits<0)
    {
      cpp_typecheck.err_location(argument1);
      throw "template argument must be greater or equal zero";
    }

    if(integer_bits>width)
    {
      cpp_typecheck.err_location(argument1);
      throw "template argument must be smaller or equal width";
    }

    dest=type_exprt(typet(base_name));
    dest.type().set("width", integer2string(width));
    dest.type().set("integer_bits", integer2string(integer_bits));
  }
  else if(base_name=="integer")
  {
    if(arguments.size()!=0)
    {
      cpp_typecheck.err_location(location);
      throw id2string(base_name)+" expects no template arguments";
    }

    dest=type_exprt(typet(base_name));
  }
  else if(has_prefix(id2string(base_name), "constant_infinity"))
  {
    // ok, but type missing
    dest=exprt("infinity");
  }
  else if(base_name=="dump_scopes")
  {
    dest=exprt("constant", typet("empty"));
    cpp_typecheck.str << "Scopes in location " << location << std::endl;
    cpp_typecheck.cpp_scopes.get_root_scope().print(cpp_typecheck.str);
    cpp_typecheck.warning();
  }
  else
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str
      << "unknown built-in identifier: " << base_name;
    throw 0;
  }

  return dest;
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::resolve_scope

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::resolve_scope(
  const cpp_namet &cpp_name,
  std::string &base_name,
  irept &template_args)
{
  assert(cpp_name.id()=="cpp-name");
  assert(!cpp_name.get_sub().empty());

  cpp_scopet& original_scope =
  cpp_typecheck.cpp_scopes.current_scope();

  const locationt &location=cpp_name.location();

  irept::subt::const_iterator pos=cpp_name.get_sub().begin();

  // check if we need to go to the root scope
  if(pos->id()=="::")
  {
    pos++;
    cpp_typecheck.cpp_scopes.go_to_root_scope();
  }

  base_name="";
  template_args.make_nil();

  while(pos!=cpp_name.get_sub().end())
  {
    if(pos->id()=="name")
      base_name+=pos->get_string("identifier");
    else if(pos->id()=="template_args")
    {
      // typecheck the template arguments in the context
      // of the original scope
      exprt tmp = static_cast<const exprt&>(*pos);
      cpp_scopet& backup_scope =
      cpp_typecheck.cpp_scopes.current_scope();
      cpp_typecheck.cpp_scopes.go_to(original_scope);

      cpp_typecheck.typecheck_template_args(tmp);
      already_typechecked(tmp);
      template_args.swap(tmp);
      cpp_typecheck.cpp_scopes.go_to(backup_scope);
    }
    else if(pos->id()=="::")
    {
      cpp_scopest::id_sett id_set;

      if(template_args.is_not_nil())
        cpp_typecheck.cpp_scopes.get_ids(base_name,cpp_idt::TEMPLATE, id_set, false);
      else
        cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, false);

      filter_for_named_scopes(id_set);


      if(id_set.empty())
      {
        cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
        cpp_typecheck.err_location(location);
        cpp_typecheck.str << "scope `" << base_name << "' not found";
        throw 0;
      }
      else if(id_set.size()==1)
      {
        if(template_args.is_not_nil())
        {
          const symbolt& symb_tmpl =
          cpp_typecheck.instantiate_template(cpp_name.location(),
                                             (*id_set.begin())->identifier, template_args);

          cpp_typecheck.cpp_scopes.go_to(
                                         cpp_typecheck.cpp_scopes.get_scope(symb_tmpl.name));
          template_args.make_nil();
        }
        else
          cpp_typecheck.cpp_scopes.go_to(**id_set.begin());
      }
      else
      {
        cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
        cpp_typecheck.err_location(location);
        cpp_typecheck.str << "scope `"
                          << base_name << "' is ambiguous";
        throw 0;
      }

      // we start from fresh
      base_name.clear();
    }
    else if(pos->id()=="operator")
    {
      base_name+="operator";

      irept::subt::const_iterator next=pos+1;
      assert(next != cpp_name.get_sub().end());

      if(next->id() == "cpp-name")
      {
        // it's a cast operator
        irept next_ir = *next;
        typet op_name;
        op_name.swap(next_ir);
        cpp_typecheck.typecheck_type(op_name);
        base_name+="("+cpp_type2name(op_name)+")";
        pos++;
      }

    }
    else
      base_name+=pos->id_string();

    pos++;
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::resolve_namespace

Inputs:

Outputs:

Purpose:

\*******************************************************************/

cpp_scopet &cpp_typecheck_resolvet::resolve_namespace(
  const cpp_namet &cpp_name)
{
  std::string base_name;
  irept template_args(get_nil_irep());

  cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);
  resolve_scope(cpp_name, base_name, template_args);

  const locationt &location=cpp_name.location();
  bool qualified=cpp_name.is_qualified();

  cpp_scopest::id_sett id_set;
  cpp_typecheck.cpp_scopes.get_ids(
                                   base_name, id_set, qualified);

  filter_for_namespaces(id_set);

  if(id_set.empty())
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str
      << "namespace `"
      << base_name << "' not found";
    throw 0;
  }
  else if(id_set.size()==1)
  {
    cpp_idt &id=**id_set.begin();
    return (cpp_scopet &)id;
  }
  else
  {
    cpp_typecheck.err_location(location);
    cpp_typecheck.str
      << "namespace `"
      << base_name << "' is ambigous";
    throw 0;
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::resolve

Inputs:

Outputs:

Purpose:

\*******************************************************************/

exprt cpp_typecheck_resolvet::resolve(
  const cpp_namet &cpp_name,
  const wantt want,
  const cpp_typecheck_fargst &fargs,
  bool fail_with_exception)
{
  std::string base_name;
  exprt template_args;
  template_args.make_nil();

  // save 'this_expr' before resolving the scopes
  this_expr=cpp_typecheck.cpp_scopes.current_scope().this_expr;

  cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

  // this changes the scope
  resolve_scope(cpp_name, base_name, template_args);

  #ifdef CPP_SYSTEMC_EXTENSION
  if(base_name == "sc_uint" || base_name == "sc_bv")
    return do_builtin_sc_uint_extension(cpp_name, template_args);
  else if(base_name == "sc_int")
    return do_builtin_sc_int_extension(cpp_name, template_args);
  else if(base_name == "sc_logic")
    return do_builtin_sc_logic_extension(cpp_name, template_args);
  else if(base_name == "sc_lv")
    return do_builtin_sc_lv_extension(cpp_name, template_args);
  else if (base_name == "SC_LOGIC_0" || base_name == "SC_LOGIC_1" ||
           base_name == "SC_LOGIC_Z" || base_name == "SC_LOGIC_X")
  {
    exprt constant("constant", typet("verilogbv"));
    constant.type().set("width", "1");
    if(base_name == "SC_LOGIC_0")
       constant.set("value", "0");
    else if(base_name == "SC_LOGIC_1")
      constant.set("value", "1");
    else if(base_name == "SC_LOGIC_Z")
      constant.set("value", "z");
    else if(base_name == "SC_LOGIC_X")
      constant.set("value", "x");
    else
      assert(0);

    if(want==TYPE)
    {
      if(!fail_with_exception) return nil_exprt();

      cpp_typecheck.err_location(cpp_name);
      cpp_typecheck.str
        << "error: expected type, but got expression `"
        << cpp_typecheck.to_string(constant) << "'";
      throw 0;
    }

    return constant;
  }
  #endif

  const locationt &location=cpp_name.location();
  bool qualified=cpp_name.is_qualified();

  // do __CPROVER scope
  if(qualified)
  {
    if(cpp_typecheck.cpp_scopes.current_scope().identifier==
       "cpp::__CPROVER")
      return do_builtin(base_name, template_args);
  }
  else
  {
    if(base_name=="true")
    {
      exprt result;
      result.make_true();
      result.location()=location;
      return result;
    }
    else if(base_name=="false")
    {
      exprt result;
      result.make_false();
      result.location()=location;
      return result;
    }
  }

  cpp_scopest::id_sett id_set;
  if(template_args.is_nil())
    cpp_typecheck.cpp_scopes.get_ids(base_name, id_set, false);
  else
    cpp_typecheck.cpp_scopes.get_ids(base_name, cpp_idt::TEMPLATE, id_set, false);

  // Argument-dependent name lookup
  if(!qualified && !fargs.has_object)
    resolve_with_arguments(id_set, base_name, fargs);

  if(id_set.empty())
  {
    if(!fail_with_exception) return nil_exprt();

    cpp_typecheck.err_location(location);
    cpp_typecheck.str
      << "symbol `"
      << base_name << "' not found";

    if(qualified)
    {
      if(cpp_typecheck.cpp_scopes.current_scope().is_root_scope())
        cpp_typecheck.str << " in root scope";
      else
        cpp_typecheck.str << " in scope `"
                          <<
          cpp_typecheck.cpp_scopes.current_scope().prefix
                          << "'";
    }

    //cpp_typecheck.cpp_scopes.get_root_scope().print(std::cout);
    throw 0;
  }

  resolve_identifierst identifiers;

  if(template_args.is_not_nil())
  {
    // methods and functions
    convert_identifiers(
      id_set, location, template_args, fargs, identifiers);

    apply_template_args(
      identifiers, template_args, fargs);
  }
  else
  {
    convert_identifiers(
      id_set, location, template_args, fargs, identifiers);
  }
  // change type into constructors if we want a constructor
  if(want==VAR)
    make_constructors(identifiers);

  filter(identifiers, want);

  exprt result;

  // We may need to disambiguate functions,
  // but don't want templates yet
  resolve_identifierst new_identifiers=identifiers;
  remove_templates(new_identifiers);

  disambiguate(new_identifiers, fargs);

  // no matches? Try again with function template guessing.
  if(new_identifiers.empty() && template_args.is_nil())
  {
    new_identifiers=identifiers;
//    guess_function_template_args(new_identifiers, fargs);
  }

  remove_duplicates(new_identifiers);

  if(new_identifiers.size()==1)
  {
    result=*new_identifiers.begin();
  }
  else
  {
    // nothing or too many
    if(!fail_with_exception) return nil_exprt();

    if(new_identifiers.empty())
    {
      cpp_typecheck.err_location(location);
      cpp_typecheck.str
        << "found no match for symbol `" << base_name
        << "', candidates are:" << std::endl;
      // TODO
//      show_identifiers(base_name, identifiers, cpp_typecheck.str);
    }
    else
    {
      cpp_typecheck.err_location(location);
      cpp_typecheck.str
        << "symbol `" << base_name
        << "' does not uniquely resolve:" << std::endl;
      // TODO
//      show_identifiers(base_name, new_identifiers, cpp_typecheck.str);
    }

    if(fargs.in_use)
    {
      cpp_typecheck.str << std::endl;
      cpp_typecheck.str << "argument types:" << std::endl;

      for(exprt::operandst::const_iterator
          it=fargs.operands.begin();
          it!=fargs.operands.end();
          it++)
      {
        cpp_typecheck.str << "  "
        		          << cpp_typecheck.to_string(it->type()) << std::endl;
      }
    }

    if(!cpp_typecheck.instantiation_stack.empty())
    {
      cpp_typecheck.str << std::endl;
      cpp_typecheck.show_instantiation_stack(cpp_typecheck.str);
    }

    throw 0;
  }

  switch(want)
  {
  case VAR:
    if(result.id()=="type" && !cpp_typecheck.cpp_is_pod(result.type()))
    {
      if(!fail_with_exception) return nil_exprt();

      cpp_typecheck.err_location(location);

      cpp_typecheck.str
        << "error: expected expression, but got type `"
        << cpp_typecheck.to_string(result.type()) << "'";

      throw 0;
    }
    break;

  case TYPE:
    if(result.id()!="type")
    {
      if(!fail_with_exception) return nil_exprt();

      cpp_typecheck.err_location(location);

      cpp_typecheck.str
        << "error: expected type, but got expression `"
        << cpp_typecheck.to_string(result) << "'";

      throw 0;
    }
    break;

  default:
	  break;
  }

  return result;
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::apply_template_args

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::apply_template_args(
  exprt &expr,
  const irept &template_args,
  const cpp_typecheck_fargst& fargs)
{
  const irept::subt &arguments=
  template_args.find("arguments").get_sub();

  typet tmp(expr.type());
  cpp_typecheck.follow_symbol(tmp);

  // template class or function?
  if(tmp.get_bool("is_template"))
  {
    if(template_args.is_nil())
    {
      expr.make_nil();
      return;
    }

    assert(tmp.id() == "cpp-declaration");

    if(tmp.find("type").id()=="struct")
    {
      if(expr.id()!="symbol")
      {
        cpp_typecheck.err_location(location);
        throw "expected template-class symbol";
      }

      const symbolt &new_symbol=
      cpp_typecheck.instantiate_template(
        location, expr.get("identifier"), template_args);

      exprt expr_type("type");
      expr_type.type().id("symbol");
      expr_type.type().set("identifier",new_symbol.name);
      expr.swap(expr_type);
    }
    else
    {
      // musst be a function, maybe method
      if(expr.id()!="symbol")
      {
        cpp_typecheck.err_location(location);
        throw "expected template function symbol type";
      }

      const symbolt &new_symbol=
      cpp_typecheck.instantiate_template(
                                         location, expr.get("identifier"), template_args);

      // check if it is a method
      const code_typet& code_type = to_code_type(new_symbol.type);
      if(!code_type.arguments().empty() && 
          code_type.arguments()[0].get("#base_name") == "this")
      {
        // do we have an object?
        if(fargs.has_object)
        {
          const symbolt& type_symb = 
            cpp_typecheck.lookup(fargs.operands.begin()->type().get("identifier"));
          const struct_typet& struct_type = to_struct_type(type_symb.type);
          assert(struct_type.has_component(new_symbol.name));
          exprt member("member",code_type);
          member.set("component_name",new_symbol.name);
          member.copy_to_operands(*fargs.operands.begin());
          member.location() = location;
          expr.swap(member);
          return;
        }

      }

      expr=cpp_symbol_expr(new_symbol);
      expr.location()=location;
    }
  }
  else if(!arguments.empty())
  {
    #if 0
    cpp_typecheck.err_location(location);
    cpp_typecheck.err
    << "expected template type left of <...>, but got "
    << cpp_typecheck.to_string(expr.type()) << std::endl;
    throw 0;
    #endif
    expr.make_nil();
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::disambiguate

Inputs:

Outputs:
/
Purpose:

\*******************************************************************/

bool cpp_typecheck_resolvet::disambiguate(
  const exprt &expr,
  unsigned &args_distance,
  const cpp_typecheck_fargst &fargs)
{
  args_distance=0;

  if(expr.type().id()!="code" || !fargs.in_use)
    return true;

  const code_typet &type=to_code_type(expr.type());

  if(expr.id()=="member" ||
     type.return_type().id() == "constructor")
  {
    // if it's a member, but does not have an object yet,
    // we add one
    if(!fargs.has_object)
    {
      const code_typet::argumentst &arguments=type.arguments();
      const code_typet::argumentt &argument = arguments.front();

      assert(argument.get("#base_name")=="this");

      if(expr.find("type").get("return_type") == "constructor")
      {
        // it's a constructor
        const typet &object_type = argument.type().subtype();
        exprt object("symbol", object_type);
        object.set("#lvalue", true);

        cpp_typecheck_fargst new_fargs(fargs);
        new_fargs.add_object(object);
        return new_fargs.match(type, args_distance, cpp_typecheck);
      }
      else
      {
        if(expr.type().get_bool("#is_operator") &&
           fargs.operands.size() == arguments.size())
        {
          return fargs.match(type, args_distance, cpp_typecheck);
        }

        cpp_typecheck_fargst new_fargs(fargs);
        new_fargs.add_object(expr.op0());
        return new_fargs.match(type, args_distance, cpp_typecheck);
      }
    }
  }
  else if(fargs.has_object)
  {
    // if it's not a member then we shall remove the object
    cpp_typecheck_fargst new_fargs(fargs);
    new_fargs.remove_object();
    return new_fargs.match(type, args_distance, cpp_typecheck);
  }


  return fargs.match(type, args_distance, cpp_typecheck);
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::filter_for_named_scopes

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::filter_for_named_scopes(
  cpp_scopest::id_sett &id_set)
{
  cpp_scopest::id_sett new_set;

  // We only want scopes!
  for(cpp_scopest::id_sett::const_iterator
      it=id_set.begin();
      it!=id_set.end();
      it++)
  {
    cpp_idt &id=**it;

    if(id.is_class() || id.is_enum() || id.is_namespace())
      new_set.insert(&id);
    else if(id.is_typedef())
    {
      irep_idt identifier=id.identifier;

      if(id.is_member)
      {
        struct_typet struct_type =
        static_cast<const struct_typet&>(cpp_typecheck.lookup(id.class_identifier).type);
        const exprt pcomp=struct_type.get_component(identifier);
        assert(pcomp.is_not_nil());
        assert(pcomp.get_bool("is_type"));
        const typet &type=pcomp.type();
        assert(type.id()!="struct");
        if(type.id()=="symbol")
          identifier = type.get("identifier");
        else
          continue;
      }

      while(true)
      {
        const symbolt &symbol=cpp_typecheck.lookup(identifier);
        assert(symbol.is_type);

        // todo? maybe do enum here, too?
        if(symbol.type.id()=="struct")
        {
          // this is a scope, too!
          cpp_idt &class_id=
          cpp_typecheck.cpp_scopes.get_id(identifier);
          new_set.insert(&class_id);
          break;
        }
        else if(symbol.type.id()=="symbol")
          identifier=symbol.type.get("identifier");
        else
          break;
      }
    }
    else if(id.id_class == cpp_scopet::TEMPLATE)
    {
      const symbolt symbol = cpp_typecheck.lookup(id.identifier);
      if(symbol.type.get("type") =="struct")
        new_set.insert(&id);
    }
    else if(id.id_class == cpp_scopet::TEMPLATE_ARGUMENT)
    {
      // maybe it is a scope
      exprt e = cpp_typecheck.template_map.lookup(id.identifier);

      if(e.id() != "type")
        continue; // it's definitively not a scope

      if(e.type().id() == "symbol")
      {
        irep_idt identifier=e.type().get("identifier");

        while(true)
        {
          const symbolt &symbol=cpp_typecheck.lookup(identifier);
          assert(symbol.is_type);

          if(symbol.type.id()=="struct")
          {
            // this is a scope, too!
            cpp_idt &class_id=
            cpp_typecheck.cpp_scopes.get_id(identifier);
            new_set.insert(&class_id);
            break;
          }
          else if(symbol.type.id()=="symbol")
            identifier=symbol.type.get("identifier");
          else
            break;
        }
      }
    }
  }

  id_set.swap(new_set);
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::filter_for_namespaces

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::filter_for_namespaces(
  cpp_scopest::id_sett &id_set)
{
  // we only want namespaces
  for(cpp_scopest::id_sett::iterator
      it=id_set.begin();
      it!=id_set.end();
      ) // no it++
  {
    if((*it)->is_namespace())
      it++;
    else
    {
      cpp_scopest::id_sett::iterator old(it);
      it++;
      id_set.erase(old);
    }
  }
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::resolve_with_arguments

Inputs:

Outputs:

Purpose:

\*******************************************************************/

void cpp_typecheck_resolvet::resolve_with_arguments(
  cpp_scopest::id_sett &id_set,
  const std::string &base_name,
  const cpp_typecheck_fargst &fargs)
{
  cpp_save_scopet save_scope(cpp_typecheck.cpp_scopes);

  for(unsigned i = 0 ; i<fargs.operands.size();i++)
  {
    const typet& final_type = cpp_typecheck.follow(fargs.operands[i].type());

    if(final_type.id() != "struct" && final_type.id() != "union")
      continue;

    cpp_scopest::id_sett tmp_set;
    cpp_idt& scope = cpp_typecheck.cpp_scopes.get_scope(final_type.get("name"));
    cpp_typecheck.cpp_scopes.go_to(scope);
    cpp_typecheck.cpp_scopes.get_ids(base_name, tmp_set, false);
    id_set.insert(tmp_set.begin(),tmp_set.end());
  }
}

#ifdef CPP_SYSTEMC_EXTENSION
/*******************************************************************\

Function: cpp_typecheck_resolvet::do_builtin_sc_uint_extension

Inputs:

Outputs:

Purpose:

\*******************************************************************/

exprt cpp_typecheck_resolvet::do_builtin_sc_uint_extension(
  const cpp_namet &cpp_name,
  exprt& template_args)
{
  if(template_args.is_nil())
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "template arguments expected";
    throw 0;
  }

  cpp_typecheck.typecheck_template_args(template_args);

  irept args=template_args.find("arguments");
  assert(args.is_not_nil() && args.get_sub().size() == 1);

  exprt arg0;
  arg0.swap(args.get_sub()[0]);

  if(!arg0.is_constant() ||
     arg0.type().id()!="signedbv")
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "integer constant expected";
    throw 0;
  }

  int width = atoi(arg0.get("#cformat").c_str());

  if(width <= 0)
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "strictly positive value expected";
    throw 0;
  }

  typet unsignedbv("unsignedbv");
  // this won't work for hex etc
  unsignedbv.add("width")=arg0.find("#cformat");

  return type_exprt(unsignedbv);
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::do_builtin_sc_int_extension

Inputs:

Outputs:

Purpose:

\*******************************************************************/

exprt cpp_typecheck_resolvet::do_builtin_sc_int_extension(
  const cpp_namet &cpp_name,
  exprt& template_args)
{
 if(template_args.is_nil())
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "template arguments expected";
    throw 0;
  }

  cpp_typecheck.typecheck_template_args(template_args);

  irept args = template_args.find("arguments");
  assert(args.is_not_nil() && args.get_sub().size() == 1);

  exprt arg0;
  arg0.swap(args.get_sub()[0]);

  if(!arg0.is_constant() || arg0.type().id() != "signedbv")
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "integer constant expected";
    throw 0;
  }

  int width = atoi(arg0.get("#cformat").c_str());

  if(width <= 0)
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "strictly positive value expected";
    throw 0;
  }

  typet unsignedbv("signedbv");
  unsignedbv.add("width") = arg0.find("#cformat");

  exprt dest=type_exprt(unsignedbv);
  dest.type().set("#sc_int", true);

  return dest;
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::do_builtin_sc_logic_extension

Inputs:

Outputs:

Purpose:

\*******************************************************************/

exprt cpp_typecheck_resolvet::do_builtin_sc_logic_extension(
  const cpp_namet &cpp_name,
  const exprt &template_args)
{
  if(template_args.is_not_nil())
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "no template argument expected";
    throw 0;
  }

  typet verilogbv("verilogbv");
  verilogbv.set("width", "1");

  return type_exprt(verilogbv);
}

/*******************************************************************\

Function: cpp_typecheck_resolvet::do_builtin_sc_lv_extension

Inputs:

Outputs:

Purpose:

\*******************************************************************/

exprt cpp_typecheck_resolvet::do_builtin_sc_lv_extension(
  const cpp_namet &cpp_name,
  exprt &template_args)
{
  if(template_args.is_nil())
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "template arguments expected";
    throw 0;
  }

  cpp_typecheck.typecheck_template_args(template_args);

  irept args = template_args.find("arguments");
  assert(args.is_not_nil() && args.get_sub().size() == 1);

  exprt arg0;
  arg0.swap(args.get_sub()[0]);

  if(!arg0.is_constant() || arg0.type().id() != "signedbv")
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "integer constant expected";
    throw 0;
  }

  int width = atoi(arg0.get("#cformat").c_str());

  if(width <= 0)
  {
    cpp_typecheck.err_location(cpp_name);
    cpp_typecheck.str << "strictly positive value expected";
    throw 0;
  }

  typet verilogbv("verilogbv");
  verilogbv.add("width") = arg0.find("#cformat");

  return type_exprt(verilogbv);
}

#endif
