#include <python-frontend/python_library.h>

#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/goto_functions.h>
#include <util/symtab/context.h>
#include <cstdlib>

extern "C"
{
  extern const uint8_t pysrc64_buf[];
  extern const unsigned int pysrc64_buf_size;
}

namespace
{
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
  for (auto &named : model_bodies.function_map)
  {
    auto it = dest.function_map.find(named.first);
    if (it == dest.function_map.end())
      continue;

    // A body the program's own conversion produced wins.
    if (it->second.body_available || named.second.body.instructions.empty())
      continue;

    it->second.body.swap(named.second.body);
    it->second.body_available = true;
  }
}
