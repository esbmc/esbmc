/*
 * clang_main.cpp
 *
 *  Created on: Jul 29, 2015
 *      Author: mramalho
 */

#include <assert.h>

#include <namespace.h>
#include <expr_util.h>
#include <std_expr.h>
#include <arith_tools.h>
#include <std_code.h>
#include <config.h>
#include <message.h>
#include <bitvector.h>
#include <c_types.h>

#include <ansi-c/c_main.h>

bool clang_main(
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
      symbolt* s = context.find_symbol(it->second);

      if(s == nullptr) continue;

      if(s->type.is_code())
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
  symbolt* s = context.find_symbol(main_symbol);
  if(s == nullptr)
    return false; // give up, no main

  const symbolt &symbol = *s;

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

      const symbolt &argc_symbol = ns.lookup("c::argc'");
      const symbolt &argv_symbol = ns.lookup("c::argv'");

      // assume argc is at least one
      exprt one = from_integer(1, argc_symbol.type);

      exprt ge(">=", bool_type());
      ge.copy_to_operands(symbol_expr(argc_symbol), one);

      init_code.copy_to_operands(code_assumet(ge));

      // assume argc is at most MAX-1
      mp_integer max;

      if(argc_symbol.type.id() == "signedbv")
        max = power(2, atoi(argc_symbol.type.width().c_str()) - 1) - 1;
      else if(argc_symbol.type.id() == "unsignedbv")
        max = power(2, atoi(argc_symbol.type.width().c_str())) - 1;
      else
        assert(false);

      exprt max_minus_one = from_integer(max - 1, argc_symbol.type);

      exprt le("<=", bool_type());
      le.copy_to_operands(symbol_expr(argc_symbol), max_minus_one);

      init_code.copy_to_operands(code_assumet(le));

      // assign argv[argc] to NULL
      constant_exprt null(
        irep_idt("NULL"),
        integer2string(0),
        argv_symbol.type.subtype());

      exprt index_expr("index", argv_symbol.type.subtype());

      index_exprt argv_index(
        symbol_expr(argv_symbol),
        symbol_expr(argc_symbol),
        argv_symbol.type.subtype());

      // disable bounds check on that one
      // Logic to perform this ^ moved into goto_check, rather than load
      // irep2 with additional baggage.

      init_code.copy_to_operands(code_assignt(argv_index, null));

      exprt::operandst &operands = call.arguments();

      if(arguments.size() == 3)
        operands.resize(3);
      else
        operands.resize(2);

      exprt &op0 = operands[0];
      exprt &op1 = operands[1];

      op0 = symbol_expr(argc_symbol);

      const exprt &arg1 = arguments[1];

      index_exprt arg1_index(
        symbol_expr(argv_symbol),
        gen_zero(index_type()),
        arg1.type().subtype());

      // disable bounds check on that one
      // Logic to perform this ^ moved into goto_check, rather than load
      // irep2 with additional baggage.

      op1 = exprt("address_of", arg1.type());
      op1.move_to_operands(arg1_index);

      if(arguments.size() == 3)
      {
        const symbolt &envp_symbol = ns.lookup("c::envp'");
        const symbolt &envp_size_symbol = ns.lookup("c::envp_size'");

        exprt envp_ge(">=", bool_type());
        envp_ge.copy_to_operands(symbol_expr(envp_size_symbol), one);

        init_code.copy_to_operands(code_assumet(envp_ge));

        // assume envp_size is at most MAX-1

        exprt envp_le("<=", bool_type());
        envp_le.copy_to_operands(symbol_expr(envp_size_symbol), max_minus_one);

        init_code.copy_to_operands(code_assumet(envp_le));

        index_exprt envp_index(
          symbol_expr(envp_symbol),
          symbol_expr(envp_size_symbol),
          envp_symbol.type.subtype());

        // disable bounds check on that one
        // Logic to perform this ^ moved into goto_check, rather than load
        // irep2 with additional baggage.

        exprt is_null("=", bool_type());
        is_null.copy_to_operands(envp_index, null);

        init_code.copy_to_operands(code_assumet(is_null));

        exprt &op2 = operands[2];
        const exprt &arg2 = arguments[2];

        index_exprt arg2_index(
          symbol_expr(envp_symbol),
          gen_zero(index_type()),
          arg2.type().subtype());

        op2 = exprt("address_of", arg2.type());
        op2.move_to_operands(arg2_index);
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



