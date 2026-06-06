#include <goto-programs/exception_globals.h>

#include <util/context.h>
#include <util/symbol.h>
#include <util/c_types.h>
#include <util/std_types.h>
#include <util/expr_util.h>

namespace
{
/// Register one zero-initialised static global, unless @p id already exists.
/// The exception state is per-thread: a propagating exception, its type and its
/// object belong to the thread that raised it, so the globals are thread-local.
/// symex routes thread-local globals to a per-thread instance (renaming.cpp), so
/// one thread cannot observe, catch, or clear another thread's in-flight
/// exception — which is what makes the lowered dispatch sound under concurrency.
void add_global(contextt &context, const char *id, const typet &type)
{
  if (context.find_symbol(id) != nullptr)
    return;

  symbolt sym;
  sym.id = id;
  sym.name = id;
  sym.mode = "C";
  sym.set_type(type);
  sym.set_value(gen_zero(type));
  sym.lvalue = true;
  sym.static_lifetime = true;
  sym.file_local = false;
  sym.is_thread_local = true;

  context.move_symbol_to_context(sym);
}
} // namespace

void create_exception_state_symbols(contextt &context)
{
  add_global(context, exception_globals::thrown_id, bool_typet());
  add_global(context, exception_globals::typeid_id, size_type());
  add_global(
    context, exception_globals::value_id, pointer_typet(empty_typet()));
}
