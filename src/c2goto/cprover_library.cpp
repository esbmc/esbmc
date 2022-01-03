/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <c2goto/cprover_library.h>
#include <cstdlib>
#include <cerrno>
#include <goto-programs/read_goto_binary.h>
#include <util/c_link.h>
#include <util/config.h>
#include <util/filesystem.h>

#ifndef NO_CPROVER_LIBRARY

extern "C"
{
  extern uint8_t clib32_buf[1];
  extern uint8_t clib64_buf[1];
  extern unsigned int clib32_buf_size;
  extern unsigned int clib64_buf_size;

  extern uint8_t clib32_fp_buf[1];
  extern uint8_t clib64_fp_buf[1];
  extern unsigned int clib32_fp_buf_size;
  extern unsigned int clib64_fp_buf_size;

  uint8_t *clib_ptrs[4][4] = {
    {&clib32_buf[0], ((&clib32_buf[0]) + clib32_buf_size)},
    {&clib64_buf[0], ((&clib64_buf[0]) + clib64_buf_size)},
    {&clib32_fp_buf[0], ((&clib32_fp_buf[0]) + clib32_fp_buf_size)},
    {&clib64_fp_buf[0], ((&clib64_fp_buf[0]) + clib64_fp_buf_size)},
  };
}

#endif

bool is_in_list(std::list<irep_idt> &list, irep_idt item)
{
  for(std::list<irep_idt>::const_iterator it = list.begin(); it != list.end();
      it++)
    if(*it == item)
      return true;

  return false;
}

void generate_symbol_deps(
  irep_idt name,
  irept irep,
  std::multimap<irep_idt, irep_idt> &deps)
{
  std::pair<irep_idt, irep_idt> type;

  if(irep.id() == "symbol")
  {
    type = std::pair<irep_idt, irep_idt>(name, irep.identifier());
    deps.insert(type);
    return;
  }

  forall_irep(irep_it, irep.get_sub())
  {
    if(irep_it->id() == "symbol")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->identifier());
      deps.insert(type);
      generate_symbol_deps(name, *irep_it, deps);
    }
    else if(irep_it->id() == "argument")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->cmt_identifier());
      deps.insert(type);
    }
    else
    {
      generate_symbol_deps(name, *irep_it, deps);
    }
  }

  forall_named_irep(irep_it, irep.get_named_sub())
  {
    if(irep_it->second.id() == "symbol")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->second.identifier());
      deps.insert(type);
    }
    else if(irep_it->second.id() == "argument")
    {
      type =
        std::pair<irep_idt, irep_idt>(name, irep_it->second.cmt_identifier());
      deps.insert(type);
    }
    else
    {
      generate_symbol_deps(name, irep_it->second, deps);
    }
  }
}

void ingest_symbol(
  irep_idt name,
  std::multimap<irep_idt, irep_idt> &deps,
  std::list<irep_idt> &to_include)
{
  std::pair<
    std::multimap<irep_idt, irep_idt>::const_iterator,
    std::multimap<irep_idt, irep_idt>::const_iterator>
    range;
  std::multimap<irep_idt, irep_idt>::const_iterator it;

  range = deps.equal_range(name);
  if(range.first == range.second)
    return;

  for(it = range.first; it != range.second; it++)
    to_include.push_back(it->second);

  deps.erase(name);
}

#ifdef NO_CPROVER_LIBRARY
void add_cprover_library(contextt &, const messaget &)
{
}

#else

static file_operations::tmp_path
dump_to_temp_file(const void *data, size_t size, const messaget &msg)
{
  using namespace file_operations;
  tmp_file tmp = create_tmp_file();
  if(fwrite(data, size, 1, tmp.file()) != 1)
  {
    msg.error(fmt::format(
      "Error writing internal C library to {}: {}",
      tmp.path(),
      strerror(errno)));
    abort();
  }
  /* Normally, ~tmp_file() would close the file, but move-constructing the
   * public base class tmp_path from it sets tmp.keep(true), which does not
   * close the file. Thus, do it here. */
  if(fclose(tmp.file()))
  {
    msg.error(fmt::format("Error closing {}: {}", tmp.path(), strerror(errno)));
    abort();
  }
  return std::move(tmp);
}

void add_cprover_library(contextt &context, const messaget &message_handler)
{
  if(config.ansi_c.lib == configt::ansi_ct::libt::LIB_NONE)
    return;

  contextt new_ctx(message_handler), store_ctx(message_handler);
  goto_functionst goto_functions;
  std::multimap<irep_idt, irep_idt> symbol_deps;
  std::list<irep_idt> to_include;
  uint8_t **this_clib_ptrs;
  uint64_t size;
  int fd;

  if(config.ansi_c.word_size == 32)
  {
    if(config.ansi_c.use_fixed_for_float)
    {
      this_clib_ptrs = &clib_ptrs[0][0];
    }
    else
    {
      this_clib_ptrs = &clib_ptrs[2][0];
    }
  }
  else if(config.ansi_c.word_size == 64)
  {
    if(config.ansi_c.use_fixed_for_float)
    {
      this_clib_ptrs = &clib_ptrs[1][0];
    }
    else
    {
      this_clib_ptrs = &clib_ptrs[3][0];
    }
  }
  else
  {
    if(config.ansi_c.word_size == 16)
    {
      message_handler.warning(
        "Warning: this version of ESBMC does not have a C library "
        "for 16 bit machines");
      return;
    }
    message_handler.error(
      fmt::format("No c library for bitwidth {}", config.ansi_c.int_width));
    abort();
  }

  size = this_clib_ptrs[1] - this_clib_ptrs[0];
  if(size == 0)
  {
    message_handler.error("error: Zero-lengthed internal C library");
    abort();
  }

  if(read_goto_binary(
       dump_to_temp_file(this_clib_ptrs[0], size, message_handler).path(),
       new_ctx,
       goto_functions,
       message_handler))
    abort();

  new_ctx.foreach_operand([&symbol_deps](const symbolt &s) {
    generate_symbol_deps(s.id, s.value, symbol_deps);
    generate_symbol_deps(s.id, s.type, symbol_deps);
  });

  // Add two hacks; we migth use either pthread_mutex_lock or the checked
  // variety; so if one version is used, pull in the other too.
  std::pair<irep_idt, irep_idt> lockcheck(
    dstring("pthread_mutex_lock"), dstring("pthread_mutex_lock_check"));
  symbol_deps.insert(lockcheck);

  std::pair<irep_idt, irep_idt> condcheck(
    dstring("pthread_cond_wait"), dstring("pthread_cond_wait_check"));
  symbol_deps.insert(condcheck);

  std::pair<irep_idt, irep_idt> joincheck(
    dstring("pthread_join"), dstring("pthread_join_noswitch"));
  symbol_deps.insert(joincheck);

  /* The code just pulled into store_ctx might use other symbols in the C
   * library. So, repeatedly search for new C library symbols that we use but
   * haven't pulled in, then pull them in. We finish when we've made a pass
   * that adds no new symbols. */

  new_ctx.foreach_operand(
    [&context, &store_ctx, &symbol_deps, &to_include](const symbolt &s) {
      const symbolt *symbol = context.find_symbol(s.id);
      if(symbol != nullptr && symbol->value.is_nil())
      {
        store_ctx.add(s);
        ingest_symbol(s.id, symbol_deps, to_include);
      }
    });

  for(std::list<irep_idt>::const_iterator nameit = to_include.begin();
      nameit != to_include.end();
      nameit++)
  {
    symbolt *s = new_ctx.find_symbol(*nameit);
    if(s != nullptr)
    {
      store_ctx.add(*s);
      ingest_symbol(*nameit, symbol_deps, to_include);
    }
  }

  if(c_link(context, store_ctx, message_handler, "<built-in-library>"))
  {
    // Merging failed
    message_handler.error("Failed to merge C library");
    abort();
  }
}
#endif
