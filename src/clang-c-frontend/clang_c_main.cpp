#include <cassert>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/config.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <clang-c-frontend/clang_c_main.h>

void clang_c_maint::init_variable(codet &dest, const symbolt &sym)
{
  const exprt &value = sym.value;

  if (value.is_nil())
    return;

  assert(!value.type().is_code());

  exprt symbol("symbol", sym.type);
  symbol.identifier(sym.id);

  code_assignt code(symbol, sym.value);
  code.location() = sym.location;

  codet adjusted("decl-block");
  adjust_init(code, adjusted);

  dest.move_to_operands(adjusted.has_operands() ? adjusted : code);
}

void clang_c_maint::adjust_init(code_assignt &, codet &)
{
  // currently nothing to be adjusted for C
}

void clang_c_maint::static_lifetime_init(const contextt &context, codet &dest)
{
  dest = code_blockt();

  // Do assignments based on "value".
  context.foreach_operand_in_order([&dest, this](const symbolt &s) {
    if (s.static_lifetime)
      init_variable(dest, s);
  });

  // call designated "initialization" functions
  context.foreach_operand_in_order([&dest](const symbolt &s) {
    if (s.type.initialization() && s.type.is_code())
    {
      code_function_callt function_call;
      function_call.function() = symbol_expr(s);
      dest.move_to_operands(function_call);
    }
  });
}

bool clang_c_maint::clang_main()
{
  irep_idt main_symbol;

  std::string main = (config.main != "") ? config.main : "main";

  // find main symbol
  std::list<irep_idt> matches;

  forall_symbol_base_map (it, context.symbol_base_map, main)
  {
    // if user provided class/contract name
    if (!config.cname.empty() && !config.main.empty())
    {
      // Allow user to provide a range. e.g.
      // --class "A B" --function test
      // --contract "Base Derieve" --function _ESBMC_main
      // yet only the first matched one will get verified
      std::istringstream iss(config.cname);
      std::string tgt_cname;
      bool is_found_sym = false;
      while (iss >> tgt_cname)
      {
        const std::string fmt = "@" + tgt_cname + "@F@" + main;
        if (it->second.as_string().find(fmt) != std::string::npos)
        {
          is_found_sym = true;
          break;
        }
      }
      if (is_found_sym)
      {
        symbolt *s = context.find_symbol(it->second);
        if (s != nullptr && s->type.is_code())
          matches.push_back(it->second);
        break; // prevent ambiguous
      }
    }
    else
    {
      // look it up
      symbolt *s = context.find_symbol(it->second);
      if (s != nullptr && s->type.is_code())
        matches.push_back(it->second);
    }
  }

  if (matches.empty())
  {
    log_error("main symbol `{}' not found", main);
    return true; // give up
  }

  if (matches.size() >= 2)
  {
    log_error("main symbol `{}' is ambiguous", main);
    return true;
  }

  main_symbol = matches.front();

  // look it up
  symbolt *s = context.find_symbol(main_symbol);
  if (s == nullptr)
    return false; // give up, no main

  const symbolt &symbol = *s;

  // check if it has a body
  if (symbol.value.is_nil())
    return false; // give up

  codet init_code;

  static_lifetime_init(context, init_code);

  init_code.make_block();
  init_code.end_location(symbol.value.end_location());

  // build call to function

  code_function_callt call;
  call.location() = symbol.location;
  call.function() = symbol_expr(symbol);

  const code_typet::argumentst &arguments =
    to_code_type(symbol.type).arguments();

  if (symbol.name == "main")
  {
    if (arguments.size() == 0)
    {
      // ok
    }
    else if (arguments.size() == 2 || arguments.size() == 3)
    {
      namespacet ns(context);

      const symbolt &argc_symbol = *ns.lookup("argc'");
      const symbolt &argv_symbol = *ns.lookup("argv'");

      exprt one = from_integer(1, argc_symbol.type);

      exprt add("+", uint_type());
      add.copy_to_operands(symbol_expr(argc_symbol), one); // argc + 1

      exprt mult("*", uint_type());
      exprt char_pointer_size = from_integer(
        config.ansi_c.pointer_width() / config.ansi_c.char_width, uint_type());
      mult.copy_to_operands(
        add, char_pointer_size); // (argc + 1) * sizeof(char *)

      // assume argc is at least one
      exprt ge(">=", bool_type());
      ge.copy_to_operands(symbol_expr(argc_symbol), one);

      init_code.copy_to_operands(code_assumet(ge));

      // assume argc is at most MAX-1
      BigInt max;

      if (argc_symbol.type.id() == "signedbv")
        max = power(2, atoi(argc_symbol.type.width().c_str()) - 1) - 1;
      else if (argc_symbol.type.id() == "unsignedbv")
        max = power(2, atoi(argc_symbol.type.width().c_str())) - 1;
      else
        assert(false);

      // The argv array of MAX elements of pointer type has to fit
      max /= config.ansi_c.pointer_width() / 8;

      exprt max_minus_one = from_integer(max - 1, argc_symbol.type);

      exprt le("<=", bool_type());
      le.copy_to_operands(symbol_expr(argc_symbol), max_minus_one);

      init_code.copy_to_operands(code_assumet(le));

      // Adjust __ESBMC_alloc_size as argv is handled as dynamic array.
      exprt dynamic_size("dynamic_size", size_type());
      dynamic_size.copy_to_operands(gen_address_of(symbol_expr(argv_symbol)));
      init_code.copy_to_operands(code_assignt(dynamic_size, mult));

      // Model argv per C11 §5.1.2.2.1: argv is left as a nondet array and
      // two assumptions encode the standard's guarantees:
      //   1. argv[argc] == NULL   (null terminator)
      //   2. argv[i]   != NULL   for every 0 <= i < argc
      constant_exprt null(
        irep_idt("NULL"), integer2string(0), argv_symbol.type.subtype());

      // 1. argv[argc] == NULL
      index_exprt argv_argc(
        symbol_expr(argv_symbol),
        symbol_expr(argc_symbol),
        argv_symbol.type.subtype());
      exprt term_eq("=", bool_type());
      term_eq.copy_to_operands(argv_argc, null);
      init_code.copy_to_operands(code_assumet(term_eq));

      // 2. forall i. (0 <= i < argc) => argv[i] != NULL
      // Encoded as the implication's contrapositive:
      //   NOT(0 <= i AND i < argc) OR argv[i] != NULL.
      // The lower-bound guard (i >= 0) is required so that the array-bounds
      // check on argv[i] inside the quantifier body can be discharged.
      symbolt argv_idx_sym;
      argv_idx_sym.name = "__ESBMC_argv_idx";
      argv_idx_sym.id = "c:@__ESBMC_argv_idx";
      argv_idx_sym.type = argc_symbol.type;
      argv_idx_sym.lvalue = true;
      symbolt *argv_idx = nullptr;
      context.move(argv_idx_sym, argv_idx);

      exprt idx = symbol_expr(*argv_idx);

      index_exprt argv_i(
        symbol_expr(argv_symbol), idx, argv_symbol.type.subtype());
      exprt ne_null("notequal", bool_type());
      ne_null.copy_to_operands(argv_i, null);

      exprt ge_zero(">=", bool_type());
      ge_zero.copy_to_operands(idx, gen_zero(argc_symbol.type));
      exprt lt_argc("<", bool_type());
      lt_argc.copy_to_operands(idx, symbol_expr(argc_symbol));
      exprt in_range("and", bool_type());
      in_range.copy_to_operands(ge_zero, lt_argc);
      exprt not_in_range("not", bool_type());
      not_in_range.copy_to_operands(in_range);
      exprt forall_body("or", bool_type());
      forall_body.copy_to_operands(not_in_range, ne_null);

      exprt addr_idx("address_of", pointer_typet(argc_symbol.type));
      addr_idx.copy_to_operands(idx);
      exprt argv_valid("forall", bool_type());
      argv_valid.copy_to_operands(addr_idx, forall_body);
      init_code.copy_to_operands(code_assumet(argv_valid));

      // Provide dereferenceable backing for argv[0] and argv[1] so programs
      // that read those strings don't trigger spurious dereference failures.
      typet char_t = argv_symbol.type.subtype().subtype();
      typet char_arr_t = array_typet(char_t, from_integer(32, index_type()));

      for (int ai = 0; ai < 2; ++ai)
      {
        char suffix = static_cast<char>('0' + ai);
        std::string sname = std::string("__ESBMC_argv_str_") + suffix;

        symbolt str_sym;
        str_sym.name = irep_idt(sname);
        str_sym.id = irep_idt("c:@" + sname);
        str_sym.type = char_arr_t;
        str_sym.static_lifetime = true;
        str_sym.lvalue = true;
        symbolt *str_ptr = nullptr;
        context.move(str_sym, str_ptr);

        exprt cond_lt("<", bool_type());
        cond_lt.copy_to_operands(
          from_integer(ai, argc_symbol.type), symbol_expr(argc_symbol));

        index_exprt argv_ai(
          symbol_expr(argv_symbol),
          from_integer(ai, index_type()),
          argv_symbol.type.subtype());

        index_exprt str_first(
          symbol_expr(*str_ptr), gen_zero(index_type()), char_t);
        exprt addr_str0("address_of", argv_symbol.type.subtype());
        addr_str0.copy_to_operands(str_first);

        code_ifthenelset if_assign;
        if_assign.cond() = cond_lt;
        if_assign.then_case() = code_assignt(argv_ai, addr_str0);
        init_code.copy_to_operands(if_assign);
      }

      exprt::operandst &operands = call.arguments();

      if (arguments.size() == 3)
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

      if (arguments.size() == 3)
      {
        const symbolt &envp_symbol = *ns.lookup("envp'");
        const symbolt &envp_size_symbol = *ns.lookup("envp_size'");

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
    call.arguments().resize(
      arguments.size(), static_cast<const exprt &>(get_nil_irep()));
  }

  // Call to main symbol is now in "call"; construct calls to thread library
  // hooks for main thread start and main thread end.

  code_function_callt thread_start_call;
  thread_start_call.location() = symbol.location;
  thread_start_call.function() =
    symbol_exprt("c:@F@__ESBMC_pthread_start_main_hook");
  code_function_callt thread_end_call;
  thread_end_call.location() = symbol.location;
  thread_end_call.function() =
    symbol_exprt("c:@F@__ESBMC_pthread_end_main_hook");

  code_function_callt atexit_call;
  atexit_call.location() = symbol.location;
  atexit_call.function() = symbol_exprt("c:@F@__ESBMC_atexit_handler");

  init_code.move_to_operands(thread_start_call);
  init_code.move_to_operands(call);
  init_code.move_to_operands(atexit_call);
  init_code.move_to_operands(thread_end_call);

  // add "main"
  symbolt new_symbol;

  code_typet main_type;
  main_type.return_type() = empty_typet();

  new_symbol.id = "__ESBMC_main";
  new_symbol.name = "__ESBMC_main";
  new_symbol.type.swap(main_type);
  new_symbol.value.swap(init_code);

  if (context.move(new_symbol))
  {
    log_error("main already defined by another language module");
    return true;
  }

  return false;
}
