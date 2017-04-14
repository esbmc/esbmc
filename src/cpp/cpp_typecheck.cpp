/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <algorithm>
#include <ansi-c/c_typecast.h>
#include <cpp/cpp_convert_type.h>
#include <cpp/cpp_declarator.h>
#include <cpp/cpp_typecheck.h>
#include <cpp/expr2cpp.h>
#include <util/arith_tools.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/location.h>
#include <util/symbol.h>

/*******************************************************************\

Function: cpp_typecheckt::this_struct_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const struct_typet &cpp_typecheckt::this_struct_type()
{
  const exprt &this_expr=
    cpp_scopes.current_scope().this_expr;

  assert(this_expr.is_not_nil());
  assert(this_expr.type().id()=="pointer");

  const typet &t=follow(this_expr.type().subtype());
  return to_struct_type(t);
}

/*******************************************************************\

Function: cpp_identifier_prefix

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_identifier_prefix(const irep_idt &mode)
{
  if(mode=="C++")
    return "cpp";
  else if(mode=="C")
    return "c";
  else
    return id2string(mode);
}

/*******************************************************************\

Function: cpp_typecheckt::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::to_string(const exprt &expr)
{
  return expr2cpp(expr, *this);
}

/*******************************************************************\

Function: cpp_typecheckt::to_string

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string cpp_typecheckt::to_string(const typet &type)
{
  return type2cpp(type, *this);
}

/*******************************************************************\

Function: cpp_typecheckt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::convert(cpp_itemt &item)
{
  if(item.is_declaration())
    convert(to_cpp_declaration(item));
  else if(item.is_linkage_spec())
    convert(item.get_linkage_spec());
  else if(item.is_namespace_spec())
    convert(item.get_namespace_spec());
  else if(item.is_using())
    convert(item.get_using());
  else
  {
    err_location(item);
    throw "unknown parse-tree element: "+item.id_string();
  }
}

/*******************************************************************\

Function: cpp_typecheckt::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::typecheck()
{
  // default linkage is C++
  current_mode="C++";

  for(cpp_parse_treet::itemst::iterator
      it=cpp_parse_tree.items.begin();
      it!=cpp_parse_tree.items.end();
      it++)
    convert(*it);

  static_initialization();

  do_not_typechecked();

  clean_up();
}

/*******************************************************************\

Function: cpp_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheck(
  cpp_parse_treet &cpp_parse_tree,
  contextt &context,
  const std::string &module,
  message_handlert &message_handler)
{
  cpp_typecheckt cpp_typecheck(cpp_parse_tree, context, module, message_handler);
  return cpp_typecheck.typecheck_main();
}

/*******************************************************************\

Function: cpp_typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool cpp_typecheck(
  exprt &expr,
  message_handlert &message_handler,
  const namespacet &ns)
{
  contextt context;
  cpp_parse_treet cpp_parse_tree;

  cpp_typecheckt cpp_typecheck(cpp_parse_tree, context,
                               ns.get_context(), "", message_handler);

  try
  {
    cpp_typecheck.typecheck_expr(expr);
  }

  catch(int e)
  {
    cpp_typecheck.error();
  }

  catch(const char *e)
  {
    cpp_typecheck.error(e);
  }

  catch(const std::string &e)
  {
    cpp_typecheck.error(e);
  }

  return cpp_typecheck.get_error_found();
}

/*******************************************************************\

Function: cpp_typecheckt::static_initialization

  Inputs:

 Outputs:

 Purpose: Initialization of static objects:

 "Objects with static storage duration (3.7.1) shall be zero-initialized
 (8.5) before any other initialization takes place. Zero-initialization
 and initialization with a constant expression are collectively called
 static initialization; all other initialization is dynamic
 initialization. Objects of POD types (3.9) with static storage duration
 initialized with constant expressions (5.19) shall be initialized before
 any dynamic initialization takes place. Objects with static storage
 duration defined in namespace scope in the same translation unit and
 dynamically initialized shall be initialized in the order in which their
 definition appears in the translation unit. [Note: 8.5.1 describes the
 order in which aggregate members are initialized. The initialization
 of local static objects is described in 6.7. ]"

\*******************************************************************/

void cpp_typecheckt::static_initialization()
{
  code_blockt block_sini; // Static Initialization Block
  code_blockt block_dini; // Dynamic Initialization Block

  disable_access_control = true;

  // first do zero initialization
  context.foreach_operand(
    [this, &block_sini] (const symbolt& s)
    {
      if(!s.static_lifetime || s.mode!=current_mode)
        return;

      // it has a non-code initializer already?
      if(s.value.is_not_nil() &&
         s.value.id()!="code")
        return;

      // it's a declaration only
      if(s.is_extern)
        return;

      if(!s.lvalue)
        return;

      zero_initializer(
        cpp_symbol_expr(s),
        s.type,
        s.location,
        block_sini.operands());
    }
  );

  while(!dinis.empty())
  {
    symbolt &symbol = *context.find_symbol(dinis.front());
    dinis.pop_front();

    if(symbol.is_extern)
      continue;

    if(symbol.mode!=current_mode)
      continue;

    assert(symbol.static_lifetime);
    assert(!symbol.is_type);
    assert(symbol.type.id()!="code");

    exprt symexpr=cpp_symbol_expr(symbol);

    if(symbol.value.is_not_nil())
    {
      if(!cpp_is_pod(symbol.type))
      {
        block_dini.move_to_operands(symbol.value);
      }
      else
      {
        exprt symbexpr("symbol", symbol.type);
        symbexpr.identifier(symbol.name);

        codet code;
        code.set_statement("assign");
        code.copy_to_operands(symbexpr, symbol.value);
        code.location()=symbol.location;

        if(symbol.value.id()=="constant")
          block_sini.move_to_operands(code);
        else
          block_dini.move_to_operands(code);
      }

      // Make it nil because we do not want
      // global_init to try to initialize the
      // object
      symbol.value.make_nil();
    }
    else
    {
      exprt::operandst ops;

      codet call=
        cpp_constructor(locationt(),
          symexpr, ops);

      if(call.is_not_nil())
        block_dini.move_to_operands(call);
    }
  }

  block_sini.move_to_operands(block_dini);

  // Create the initialization procedure
  symbolt init_symbol;

  init_symbol.name="c::#ini#"+id2string(module);
  init_symbol.base_name="#ini#"+id2string(module);
  init_symbol.value.swap(block_sini);
  init_symbol.mode=current_mode;
  init_symbol.module=module;
  init_symbol.type=code_typet();
  init_symbol.type.add("return_type")=typet("empty");
  init_symbol.type.set("initialization", true);
  init_symbol.is_type=false;
  init_symbol.is_macro=false;

  context.move(init_symbol);

  disable_access_control=false;
}

/*******************************************************************\

Function: cpp_typecheckt::do_not_typechecked

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::do_not_typechecked()
{
  bool cont;

  do
  {
    cont = false;

    std::vector<symbolt*> to_typecheck_list;

    context.Foreach_operand(
      [&to_typecheck_list] (symbolt& s)
      {
        if(s.value.id()=="cpp_not_typechecked" &&
           s.value.get_bool("is_used"))
        {
          assert(s.type.is_code());
          to_typecheck_list.push_back(&s);
        }
      }
    );

    for (symbolt *sym : to_typecheck_list) {
      if (sym->base_name =="operator=")
      {
        cpp_declaratort declarator;
        declarator.location() = sym->location;
        default_assignop_value(
          lookup(sym->type.get("#member_name")),declarator);
        sym->value.swap(declarator.value());
        convert_function(*sym);
        cont=true;
      }
      else if (sym->value.operands().size() == 1)
      {
        exprt tmp = sym->value.operands()[0];
        sym->value.swap(tmp);
        convert_function(*sym);
        cont=true;
      }
      else
        assert(0); // Don't know what to do!
    }
  }
  while(cont);

  context.Foreach_operand(
    [] (symbolt& s)
    {
      if(s.value.id()=="cpp_not_typechecked")
        s.value.make_nil();
    }
  );
}

/*******************************************************************\

Function: cpp_typecheckt::clean_up

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void cpp_typecheckt::clean_up()
{
  context.Foreach_operand(
    [this] (symbolt& s)
    {
      if(s.type.get_bool("is_template"))
      {
        context.erase_symbol(s.name);
        return;
      }
      else if(s.type.is_struct() ||
              s.type.is_union())
      {
        struct_typet &struct_type = to_struct_type(s.type);

        const struct_typet::componentst &components =
          struct_type.components();

        struct_typet::componentst data_members;
        data_members.reserve(components.size());

        struct_typet::componentst &function_members =
          struct_type.methods();

        function_members.reserve(components.size());

        for(struct_typet::componentst::const_iterator
            compo_it = components.begin();
            compo_it != components.end();
            compo_it++)
        {
          if(compo_it->get_bool("is_static") ||
             compo_it->is_type())
          {
            // skip it
          }
          else if(compo_it->type().id()=="code")
          {
            function_members.push_back(*compo_it);
          }
          else
          {
            data_members.push_back(*compo_it);
          }
        }

        struct_type.components().swap(data_members);
      }
    }
  );
}
