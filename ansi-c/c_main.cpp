/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <namespace.h>
#include <expr_util.h>
#include <std_expr.h>
#include <arith_tools.h>
#include <std_code.h>
#include <config.h>
#include <type_byte_size.h>

#include "c_types.h"
#include "c_main.h"

/*******************************************************************\

Function: static_lifetime_init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

static void
scan_for_union_init(exprt &value, const expr2tc &access, std::list<std::pair<expr2tc, expr2tc> > &hoisted_unions)
{

  if (value.is_constant() && value.type().is_union()) {
    // OK -- this is an assignment of a union constant / literal to a value of
    // union type. This must be removed. We replace with a constant array of
    // zeros, because later type rewriting in goto_convert_functionst will
    // resolve the storage for this constant union to being an array.
    // The actual data in the union literal is hoisted out, and made a problem
    // for a different function.
    expr2tc union_value;
    migrate_expr(value, union_value);
    hoisted_unions.push_back(std::make_pair(access, union_value));

    // Produce array of bytes.
    BigInt size = type_byte_size(*union_value->type);
    type2tc arraytype(new array_type2t(get_uint8_type(),
          gen_ulong(size.to_uint64()), false));
    constant_int2tc zerobyte(get_uint8_type(), BigInt(0));
    constant_array_of2tc arrayof(arraytype, zerobyte);
    value = migrate_expr_back(arrayof);

    // Cease recursion
    return;
  } else {
    for (unsigned int i = 0; i < value.operands().size(); i++) {
      // Build access expression. Construct in irep2 for type sanity. First
      // convert existing expressions,
      exprt &operand = value.operands()[i];
      expr2tc new_access, operand_expr2;
      type2tc new_type;

      migrate_expr(operand, operand_expr2);
      migrate_type(operand.type(), new_type);

      // Construct an expression accessing the field that we're about to scan
      if (value.type().is_array()) {
        new_access = index2tc(new_type, operand_expr2, gen_ulong(i));
      } else if (value.type().is_struct()) {
        expr2tc value_expr2;
        migrate_expr(value, value_expr2);
        const struct_type2t &struct_type = to_struct_type(value_expr2->type);
        new_access =
          member2tc(new_type, operand_expr2, struct_type.member_names[i]);
      } else {
        std::cerr << "Unrecognized type " << value.type().id() << " when ";
        std::cerr << "rewriting union initialization" << std::endl;
        abort();
      }

      // Continue scanning for union initialization
      scan_for_union_init(operand, new_access, hoisted_unions);
    }
  }
}

static void
init_variable(codet &dest, const symbolt &sym)
{
  const exprt &value = sym.value;

  if(value.is_nil())
    return;

  assert(!value.type().is_code());

  exprt symbol("symbol", sym.type);
  symbol.identifier(sym.name);

  code_assignt code(symbol, sym.value);
  code.location() = sym.location;

  dest.move_to_operands(code);
}

void static_lifetime_init(
  const contextt &context,
  codet &dest)
{
  dest=code_blockt();

  // Do assignments based on "value".
  forall_symbols(it, context.symbols)
    if(it->second.static_lifetime)
      init_variable(dest, it->second);

  // call designated "initialization" functions

  forall_symbols(it, context.symbols)
  {
    if(it->second.type.initialization() &&
       it->second.type.is_code())
    {
      code_function_callt function_call;
      function_call.function()=symbol_expr(it->second);
      dest.move_to_operands(function_call);
    }
  }
}

/*******************************************************************\

Function: c_main

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool c_main(
  contextt &context,
  const std::string &default_prefix __attribute__((unused)),
  const std::string &standard_main,
  message_handlert &message_handler)
{
  irep_idt main_symbol;

  // find main symbol
  if(config.main!="")
  {
    std::list<irep_idt> matches;

    forall_symbol_base_map(it, context.symbol_base_map, config.main)
    {
      // look it up
      symbolst::const_iterator s_it=context.symbols.find(it->second);

      if(s_it==context.symbols.end()) continue;

      if(s_it->second.type.is_code())
        matches.push_back(it->second);
    }

    if(matches.empty())
    {
      messaget message(message_handler);
      message.error("main symbol `"+config.main+"' not found");
      return true; // give up
    }


    if(matches.size()>=2)
    {
      messaget message(message_handler);
      if (matches.size()==2)
        std::cerr << "warning: main symbol `" << config.main << "' is ambiguous" << std::endl;
      else
      {
    	message.error("main symbol `"+config.main+"' is ambiguous");
        return true;
      }
    }

    main_symbol=matches.front();

  }
  else
    main_symbol=standard_main;

  // look it up
  symbolst::const_iterator s_it=context.symbols.find(main_symbol);

  if(s_it==context.symbols.end())
    return false; // give up, no main

  const symbolt &symbol=s_it->second;

  // check if it has a body
  if(symbol.value.is_nil())
    return false; // give up

  codet init_code;

  static_lifetime_init(context, init_code);

  init_code.make_block();

  // build call to function

  code_function_callt call;
  call.location()=symbol.location;
  call.function()=symbol_expr(symbol);

  const code_typet::argumentst &arguments=
    to_code_type(symbol.type).arguments();

  if(symbol.name==standard_main)
  {
    if(arguments.size()==0)
    {
      // ok
    }
    else if(arguments.size()==2 || arguments.size()==3)
    {
      namespacet ns(context);

      const symbolt &argc_symbol=ns.lookup("c::argc'");
      const symbolt &argv_symbol=ns.lookup("c::argv'");

      {
        // assume argc is at least one
        exprt one=from_integer(1, argc_symbol.type);

        exprt ge(">=", typet("bool"));
        ge.copy_to_operands(symbol_expr(argc_symbol), one);

        codet assumption;
        assumption.set_statement("assume");
        assumption.move_to_operands(ge);
        init_code.move_to_operands(assumption);
      }

      {
        // assume argc is at most MAX-1
        mp_integer max;

        if(argc_symbol.type.id()=="signedbv")
          max=power(2, atoi(argc_symbol.type.width().c_str())-1)-1;
        else if(argc_symbol.type.id()=="unsignedbv")
          max=power(2, atoi(argc_symbol.type.width().c_str()))-1;
        else
          assert(false);

        exprt max_minus_one=from_integer(max-1, argc_symbol.type);

        exprt ge("<=", typet("bool"));
        ge.copy_to_operands(symbol_expr(argc_symbol), max_minus_one);

        codet assumption;
        assumption.set_statement("assume");
        assumption.move_to_operands(ge);
        init_code.move_to_operands(assumption);
      }

      if(arguments.size()==3)
      {
        const symbolt &envp_size_symbol=ns.lookup("c::envp_size'");
        // assume envp_size is at most MAX-1
        mp_integer max;

        if(envp_size_symbol.type.id()=="signedbv")
          max=power(2, atoi(envp_size_symbol.type.width().c_str())-1)-1;
        else if(envp_size_symbol.type.id()=="unsignedbv")
          max=power(2, atoi(envp_size_symbol.type.width().c_str()))-1;
        else
          assert(false);

        exprt max_minus_one=from_integer(max-1, envp_size_symbol.type);

        exprt ge("<=", typet("bool"));
        ge.copy_to_operands(symbol_expr(envp_size_symbol), max_minus_one);

        codet assumption;
        assumption.set_statement("assume");
        assumption.move_to_operands(ge);
        init_code.move_to_operands(assumption);
      }

      {
        /* zero_string doesn't work yet */

        /*
        exprt zero_string("zero_string", array_typet());
        zero_string.type().subtype()=char_type();
        zero_string.type().size("infinity");
        exprt index("index", char_type());
        index.copy_to_operands(zero_string, gen_zero(uint_type()));
        exprt address_of("address_of", pointer_typet());
        address_of.type().subtype()=char_type();
        address_of.copy_to_operands(index);

        if(argv_symbol.type.subtype()!=address_of.type())
          address_of.make_typecast(argv_symbol.type.subtype());

        // assign argv[*] to the address of a string-object
        exprt array_of("array_of", argv_symbol.type);
        array_of.copy_to_operands(address_of);

        init_code.copy_to_operands(
          code_assignt(symbol_expr(argv_symbol), array_of));
        */
      }

      {
        // assign argv[argc] to NULL
        exprt null("constant", argv_symbol.type.subtype());
        null.value("NULL");

        exprt index_expr("index", argv_symbol.type.subtype());
        index_expr.copy_to_operands(
          symbol_expr(argv_symbol),
          symbol_expr(argc_symbol));

        // disable bounds check on that one
        // Logic to perform this ^ moved into goto_check, rather than load
        // irep2 with additional baggage.

        init_code.copy_to_operands(code_assignt(index_expr, null));
      }

      if(arguments.size()==3)
      {
        const symbolt &envp_symbol=ns.lookup("c::envp'");
        const symbolt &envp_size_symbol=ns.lookup("c::envp_size'");

        // assume envp[envp_size] is NULL
        exprt null("constant", envp_symbol.type.subtype());
        null.value("NULL");

        exprt index_expr("index", envp_symbol.type.subtype());
        index_expr.copy_to_operands(
          symbol_expr(envp_symbol),
          symbol_expr(envp_size_symbol));

        // disable bounds check on that one
        // Logic to perform this ^ moved into goto_check, rather than load
        // irep2 with additional baggage.

        exprt is_null("=", typet("bool"));
        is_null.copy_to_operands(index_expr, null);

        codet assumption2;
        assumption2.set_statement("assume");
        assumption2.move_to_operands(is_null);
        init_code.move_to_operands(assumption2);
      }

      {
        exprt::operandst &operands=call.arguments();

        if(arguments.size()==3)
          operands.resize(3);
        else
          operands.resize(2);

        exprt &op0=operands[0];
        exprt &op1=operands[1];

        op0=symbol_expr(argc_symbol);

        {
          const exprt &arg1=arguments[1];

          exprt index_expr("index", arg1.type().subtype());
          index_expr.copy_to_operands(symbol_expr(argv_symbol), gen_zero(index_type()));

          // disable bounds check on that one
          // Logic to perform this ^ moved into goto_check, rather than load
          // irep2 with additional baggage.

          op1=exprt("address_of", arg1.type());
          op1.move_to_operands(index_expr);
        }

        // do we need envp?
        if(arguments.size()==3)
        {
          const symbolt &envp_symbol=ns.lookup("c::envp'");
          exprt &op2=operands[2];

          const exprt &arg2=arguments[2];

          exprt index_expr("index", arg2.type().subtype());
          index_expr.copy_to_operands(
            symbol_expr(envp_symbol), gen_zero(index_type()));

          op2=exprt("address_of", arg2.type());
          op2.move_to_operands(index_expr);
        }
      }
    }
    else
      assert(false);
  }
  else
  {
    call.arguments().resize(arguments.size(), static_cast<const exprt &>(get_nil_irep()));
  }

  // Call to main symbol is now in "call"; construct calls to thread library
  // hooks for main thread start and main thread end.

  code_function_callt thread_start_call;
  thread_start_call.location()=symbol.location;
  thread_start_call.function()=symbol_exprt("c::pthread_start_main_hook");
  code_function_callt thread_end_call;
  thread_end_call.location()=symbol.location;
  thread_end_call.function()=symbol_exprt("c::pthread_end_main_hook");

  init_code.move_to_operands(thread_start_call);
  init_code.move_to_operands(call);
  init_code.move_to_operands(thread_end_call);

  // add "main"
  symbolt new_symbol;

  code_typet main_type;
  main_type.return_type()=empty_typet();

  new_symbol.name="main";
  new_symbol.type.swap(main_type);
  new_symbol.value.swap(init_code);

  if(context.move(new_symbol))
  {
    messaget message;
    message.set_message_handler(&message_handler);
    message.error("main already defined by another language module");
    return true;
  }

  return false;
}
