#include <goto-programs/goto_convert_class.h>
#include <util/rename.h>

void goto_convertt::new_name(symbolt &symbol)
{
  // rename it
  get_new_name(symbol, ns);

  // store in context
  context.add(symbol);
}

void goto_convert(
  const codet &code,
  contextt &context,
  optionst &options,
  goto_programt &dest)
{
  goto_convertt goto_convert(context, options);
  goto_convert.goto_convert(code, dest);
}

void goto_convert(contextt &context, optionst &options, goto_programt &dest)
{
  // find main symbol
  const symbolt *s = context.find_symbol("__ESBMC_main");
  if(s == nullptr)
    throw "failed to find main symbol";

  log_status("goto_convert : start converting symbol table to goto functions ");
  ::goto_convert(to_code(s->value), context, options, dest);
}
