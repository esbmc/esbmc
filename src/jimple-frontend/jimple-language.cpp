#include <assert.h>
#include <jimple-frontend/jimple-language.h>

#include <c2goto/cprover_library.h>

languaget *new_jimple_language()
{
  return new jimple_languaget;
}

bool jimple_languaget::final(contextt &context)
{
  log_status("Adding cprover library");
  add_cprover_library(context);

  add_intrinsics(context);
  log_status("Adding __ESBMC_main");
  setup_main(context);
  return false;
}

bool jimple_languaget::from_type(
  const typet &,
  std::string &,
  const namespacet &,
  unsigned)
{
  // TODO
  assert(!"Not implemented yet");
  return false;
}

bool jimple_languaget::from_expr(
  const exprt &,
  std::string &,
  const namespacet &,
  unsigned)
{
  // TODO
  assert(!"Not implemented yet");
  return false;
}

unsigned jimple_languaget::default_flags(presentationt) const
{
  // TODO
  assert(!"Not implemented yet");
  return 0;
}

static inline void init_variable(codet &dest, const symbolt &sym)
{
  const exprt &value = sym.get_value();

  if (value.is_nil())
    return;

  assert(!value.type().is_code());

  exprt symbol("symbol", sym.get_type());
  symbol.identifier(sym.id);

  code_assignt code(symbol, sym.get_value());
  code.location() = sym.location;

  dest.move_to_operands(code);
}

static inline void static_lifetime_init(const contextt &context, codet &dest)
{
  dest = code_blockt();

  // Do assignments based on "value".
  context.foreach_operand_in_order([&dest](const symbolt &s) {
    if (s.static_lifetime)
      init_variable(dest, s);
  });

  // call designated "initialization" functions
  context.foreach_operand_in_order([&dest](const symbolt &s) {
    if (s.get_type().initialization() && s.get_type().is_code())
    {
      code_function_callt function_call;
      function_call.function() = symbol_expr(s);
      dest.move_to_operands(function_call);
    }
  });
}

static void add_global_static_variable(
  contextt &ctx,
  const type2tc &t,
  const std::string &name)
{
  // TODO: Maybe they should be part of Jimple context?
  std::string id = "c:@" + name;
  symbolt symbol;
  symbol.mode = "C";
  symbol.set_type(t);
  symbol.name = name;
  symbol.id = id;

  symbol.lvalue = true;
  symbol.static_lifetime = true;
  symbol.is_extern = false;
  symbol.file_local = false;
  // Drops the legacy `#zero_initializer` marker, which no IREP2 node models and
  // only Solidity's converter reads (#4715).
  symbol.set_value(gen_zero(t, true));

  ctx.move_symbol_to_context(symbol);
}

void jimple_languaget::add_intrinsics(contextt &context)
{
  const type2tc type1 = array_type2tc(get_bool_type(), expr2tc(), true);
  add_global_static_variable(context, type1, "__ESBMC_alloc");
  add_global_static_variable(context, type1, "__ESBMC_is_dynamic");

  const type2tc type2 = array_type2tc(size_type2(), expr2tc(), true);
  add_global_static_variable(context, type2, "__ESBMC_alloc_size");

  add_global_static_variable(context, int_type2(), "__ESBMC_rounding_mode");
}

void jimple_languaget::setup_main(contextt &context)
{
  irep_idt main_symbol;

  std::string main =
    (config.main != "") ? config.main : "main_0"; // main(String[])

  // find main symbol
  std::list<irep_idt> matches;

  context.foreach_named_symbol_with_base(main, [&](irep_idt id) {
    // look it up
    symbolt *s = context.find_symbol(id);
    if (s != nullptr && s->get_type().is_code())
      matches.push_back(id);
    return true; // collect all; ambiguity is reported after the scan
  });
  if (matches.empty())
    abort();

  main_symbol = matches.front();

  // look it up
  symbolt *s = context.find_symbol(main_symbol);
  if (s == nullptr)
  {
    log_error("No main method");
    abort();
    return; // give up, no main
  }

  const symbolt &symbol = *s;
  // check if it has a body
  if (symbol.get_value().is_nil())
  {
    log_error("Empty body for main");
    abort();
  }

  codet init_code;
  static_lifetime_init(context, init_code);

  init_code.make_block();

  // build call to function

  code_function_callt call;
  call.function() = symbol_expr(symbol);

  const code_typet::argumentst &arguments =
    to_code_type(symbol.get_type()).arguments();

  call.arguments().resize(
    arguments.size(), static_cast<const exprt &>(get_nil_irep()));

  // TODO: Add Threads?
  init_code.move_to_operands(call);

  // add "main"
  symbolt new_symbol;

  new_symbol.id = "__ESBMC_main";
  new_symbol.name = "__ESBMC_main";
  new_symbol.set_type(code_type2tc(
    std::vector<type2tc>{},
    get_empty_type(),
    std::vector<irep_idt>{},
    /*ellipsis=*/false));
  expr2tc init_code2;
  migrate_expr(init_code, init_code2);
  new_symbol.set_value(init_code2);

  if (context.move(new_symbol))
  {
    log_error("main already defined by another language module");
    return;
  }

  return;
}
