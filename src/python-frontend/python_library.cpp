#include <python-frontend/python_library.h>

#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>
#include <util/message/message.h>
#include <cstdlib>

extern "C"
{
  extern const uint8_t pysrc64_buf[];
  extern const unsigned int pysrc64_buf_size;
}

namespace
{
/// The blob is process-wide and read once, so its bodies wait here between
/// add_cpython_library (during typecheck, which has no goto_functionst) and
/// link_cpython_library_bodies (after goto_convert, which does).
goto_functionst model_bodies;
} // namespace

void add_cpython_library(contextt &context)
{
  if (pysrc64_buf_size == 0)
    return;

  contextt models_ctx, ignored_ctx;
  goto_binary_reader reader;
  if (reader.read_goto_binary_array(
        pysrc64_buf, pysrc64_buf_size, models_ctx, ignored_ctx, &model_bodies))
    abort();

  models_ctx.foreach_operand([&context](const symbolt &s) {
    if (context.find_symbol(s.id) == nullptr)
      context.add(const_cast<symbolt &>(s));
  });
}

void link_cpython_library_bodies(goto_functionst &dest)
{
  unsigned linked = 0, missing = 0;
  for (auto &named : model_bodies.function_map)
  {
    auto it = dest.function_map.find(named.first);
    if (it == dest.function_map.end())
    {
      ++missing;
      continue;
    }

    // goto_convert already set the type from the declaration; only the body
    // is missing, and a body the program's own conversion produced wins.
    if (it->second.body_available)
      continue;

    if (named.second.body.instructions.empty())
      continue;

    it->second.body.swap(named.second.body);
    it->second.body_available = true;
    ++linked;
  }

  log_debug(
    "python",
    "model bodies: {} of {} linked, {} had no declaration",
    linked,
    model_bodies.function_map.size(),
    missing);
}
