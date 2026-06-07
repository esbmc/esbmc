#include <goto-programs/exception_globals.h>

#include <util/context.h>
#include <util/symbol.h>
#include <util/c_types.h>
#include <util/std_types.h>
#include <util/expr_util.h>

namespace
{
/// Register one zero-initialised static global.
/// The exception state is per-thread: a propagating exception, its type and its
/// object belong to the thread that raised it, so the globals are thread-local.
/// symex routes thread-local globals to a per-thread instance (renaming.cpp), so
/// one thread cannot observe, catch, or clear another thread's in-flight
/// exception — which is what makes the lowered dispatch sound under concurrency.
///
/// If @p id already exists it is a bare extern declaration (the OM
/// std::uncaught_exceptions() reads __ESBMC_exc_uncaught_count, so linking that
/// body pulls the declaration into the context before this pass runs). Keep its
/// frontend-assigned type but upgrade the storage to a real zero-initialised
/// thread-local static.
void add_global(contextt &context, const char *id, const typet &type)
{
  if (symbolt *existing = context.find_symbol(id))
  {
    existing->set_value(gen_zero(existing->get_type()));
    existing->lvalue = true;
    existing->static_lifetime = true;
    existing->file_local = false;
    existing->is_extern = false;
    existing->is_thread_local = true;
    return;
  }

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
  add_global(context, exception_globals::uncaught_count_id, size_type());
}
