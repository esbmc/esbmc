#include <assert.h>
#include <jimple-frontend/jimple-language.h>
#include <util/message/format.h>
#include <c2goto/cprover_library.h>
languaget *new_jimple_language(const messaget &msg)
{
  return new jimple_languaget(msg);
}
bool jimple_languaget::final(contextt &context, const messaget &msg)
{
  msg.status("Adding cprover library");
  add_cprover_library(context, msg);
  msg.status("Adding __ESBMC_main");
  setup_main(context);
  return false;
}
bool jimple_languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns)
{
  //code = expr2c(expr, ns);
  return false;
}
bool jimple_languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns)
{
  //code = expr2c(expr, ns);
  return false;
}

static inline void init_variable(codet &dest, const symbolt &sym)
{
  const exprt &value = sym.value;

  if(value.is_nil())
    return;

  assert(!value.type().is_code());

  exprt symbol("symbol", sym.type);
  symbol.identifier(sym.id);

  code_assignt code(symbol, sym.value);
  code.location() = sym.location;

  dest.move_to_operands(code);
}

static inline void static_lifetime_init(const contextt &context, codet &dest)
{
  dest = code_blockt();

  // Do assignments based on "value".
  context.foreach_operand_in_order([&dest](const symbolt &s) {
    if(s.static_lifetime)
      init_variable(dest, s);
  });

  // call designated "initialization" functions
  context.foreach_operand_in_order([&dest](const symbolt &s) {
    if(s.type.initialization() && s.type.is_code())
    {
      code_function_callt function_call;
      function_call.function() = symbol_expr(s);
      dest.move_to_operands(function_call);
    }
  });
}

void jimple_languaget::setup_main(contextt &context)
{
  irep_idt main_symbol;

  std::string main = (config.main != "") ? config.main : "main";

  // find main symbol
  std::list<irep_idt> matches;

  forall_symbol_base_map(it, context.symbol_base_map, main)
  {
    // look it up
    symbolt *s = context.find_symbol(it->second);

    if(s == nullptr)
      continue;

    if(s->type.is_code())
      matches.push_back(it->second);
  }
  if(matches.empty())
    abort();

  main_symbol = matches.front();

  // look it up
  symbolt *s = context.find_symbol(main_symbol);
  if(s == nullptr)
  {
    msg.error("No main asdf");
    abort();
    return; // give up, no main
  }

  const symbolt &symbol = *s;
  // check if it has a body
  if(symbol.value.is_nil())
  {
    msg.error("Empty body for main");
    abort();
  }

  codet init_code;
  static_lifetime_init(context, init_code);

  init_code.make_block();
  //init_code.end_location(symbol.value.end_location());

  // build call to function

  code_function_callt call;
  //call.location() = symbol.location;
  call.function() = symbol_expr(symbol);

  //const code_typet::argumentst &arguments =
  // to_code_type(symbol.type).arguments();

  // Call to main symbol is now in "call"; construct calls to thread library
  // hooks for main thread start and main thread end.

  //code_function_callt thread_start_call;
  //thread_start_call.location() = symbol.location;
  //thread_start_call.function() = symbol_exprt("c:@F@pthread_start_main_hook");
  //code_function_callt thread_end_call;
  //thread_end_call.location() = symbol.location;
  //thread_end_call.function() = symbol_exprt("c:@F@pthread_end_main_hook");

  //init_code.move_to_operands(thread_start_call);
  init_code.move_to_operands(call);
  //init_code.move_to_operands(thread_end_call);

  // add "main"
  symbolt new_symbol;

  code_typet main_type;
  main_type.return_type() = empty_typet();

  new_symbol.id = "__ESBMC_main";
  new_symbol.name = "__ESBMC_main";
  new_symbol.type.swap(main_type);
  new_symbol.value.swap(init_code);

  if(context.move(new_symbol))
  {
    msg.error("main already defined by another language module");
    return;
  }

  return;
}
