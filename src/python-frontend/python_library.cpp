#include <python-frontend/python_library.h>

#include <goto-programs/goto_binary_reader.h>
#include <util/symtab/context.h>
#include <cstdlib>

extern "C"
{
  extern const uint8_t pysrc64_buf[];
  extern const unsigned int pysrc64_buf_size;
}

void add_cpython_library(contextt &context)
{
  if (pysrc64_buf_size == 0)
    return;

  contextt models_ctx, ignored_ctx;
  goto_binary_reader reader;
  if (reader.read_goto_binary_array(
        pysrc64_buf, pysrc64_buf_size, models_ctx, ignored_ctx))
    abort();

  models_ctx.foreach_operand([&context](const symbolt &s) {
    if (context.find_symbol(s.id) == nullptr)
      context.add(const_cast<symbolt &>(s));
  });
}
