/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
}

#include <sstream>
#include <istream>
#include <fstream>

#include <config.h>

#include <goto-programs/read_goto_binary.h>

#include "cprover_library.h"
#include "ansi_c_language.h"

#ifndef NO_CPROVER_LIBRARY
extern uint8_t _binary_clib16_goto_start;
extern uint8_t _binary_clib32_goto_start;
extern uint8_t _binary_clib64_goto_start;
extern uint8_t _binary_clib16_goto_end;
extern uint8_t _binary_clib32_goto_end;
extern uint8_t _binary_clib64_goto_end;

uint8_t *clib_ptrs[3][2] = {
{ &_binary_clib16_goto_start, &_binary_clib16_goto_end},
{ &_binary_clib32_goto_start, &_binary_clib32_goto_end},
{ &_binary_clib64_goto_start, &_binary_clib64_goto_end},
};
#endif

void
fetch_list_of_contained_symbols(irept irep, std::set<irep_idt> &names)
{

  forall_irep(irep_it, irep.get_sub()) {
    if (irep_it->id() == "symbol") {
      names.insert(irep_it->get("identifier"));
    } else if (irep_it->id() == "argument") {
      names.insert(irep_it->get("#identifier"));
    } else {
      fetch_list_of_contained_symbols(*irep_it, names);
    }
  }

  forall_named_irep(irep_it, irep.get_named_sub()) {
    if (irep_it->second.id() == "symbol") {
      names.insert(irep_it->second.get("identifier"));
    } else if (irep_it->second.id() == "argument") {
      names.insert(irep_it->second.get("#identifier"));
    } else {
     fetch_list_of_contained_symbols(irep_it->second, names);
    }
  }

  return;
}

void add_cprover_library(
  contextt &context,
  message_handlert &message_handler)
{
#ifdef NO_CPROVER_LIBRARY
  return;
#else
  contextt new_ctx, store_ctx, remain_ctx;
  goto_functionst goto_functions;
  ansi_c_languaget ansi_c_language;
  char symname_buffer[256];
  FILE *f;
  uint8_t **this_clib_ptrs;
  unsigned long size;
  unsigned int num_syms;
  int fd;

  if(config.ansi_c.lib==configt::ansi_ct::LIB_NONE)
    return;

  if (config.ansi_c.int_width == 16) {
    this_clib_ptrs = &clib_ptrs[0][0];
  } else if (config.ansi_c.int_width == 32) {
    this_clib_ptrs = &clib_ptrs[1][0];
  } else if (config.ansi_c.int_width == 64) {
    this_clib_ptrs = &clib_ptrs[2][0];
  } else {
    std::cerr << "No c library for bitwidth " << config.ansi_c.int_width << std::endl;
    abort();
  }

  size = (unsigned long)this_clib_ptrs[1] - (unsigned long)this_clib_ptrs[0];
  if (size == 0) {
    std::cerr << "error: Zero-lengthed internal C library" << std::endl;
    abort();
  }

  sprintf(symname_buffer, "/tmp/ESBMC_XXXXXX");
  fd = mkstemp(symname_buffer);
  close(fd);
  f = fopen(symname_buffer, "w");
  if (fwrite(this_clib_ptrs[0], size, 1, f) != 1) {
    std::cerr << "Couldn't manipulate internal C library" << std::endl;
    abort();
  }
  fclose(f);

  std::ifstream infile(symname_buffer);
  read_goto_binary(infile, new_ctx, goto_functions, message_handler);
  unlink(symname_buffer);

  /* The code just pulled into store_ctx might use other symbols in the C
   * library. So, repeatedly search for new C library symbols that we use but
   * haven't pulled in, then pull them in. We finish when we've made a pass
   * that adds no new symbols. */
  forall_symbols(it, new_ctx.symbols) {
    symbolst::const_iterator used_sym = context.symbols.find(it->second.name);

    if (used_sym != context.symbols.end() && used_sym->second.value.is_nil()){
      store_ctx.add(it->second);
    } else {
      remain_ctx.add(it->second);
    }
  }

  std::set<irep_idt> names;
  do {
    num_syms = store_ctx.symbols.size();
    forall_symbols(it, store_ctx.symbols) {
      fetch_list_of_contained_symbols(it->second.value, names);
      fetch_list_of_contained_symbols(it->second.type, names);

      for (std::set<irep_idt>::const_iterator nameit = names.begin();
            nameit != names.end(); nameit++) {

        symbolst::const_iterator used_sym = new_ctx.symbols.find(*nameit);
        if (used_sym != new_ctx.symbols.end())
          store_ctx.add(used_sym->second);
      }
    }
  } while (num_syms != store_ctx.symbols.size());

  ansi_c_language.merge_context(
      context, store_ctx, message_handler, "<built-in-library>");

#endif
}
