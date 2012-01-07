/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#undef small // MinGW headers are terrible (alternately; windows).
#endif
}

#include <sstream>
#include <istream>
#include <fstream>

#include <config.h>

#include <goto-programs/read_goto_binary.h>

#include "cprover_library.h"
#include "ansi_c_language.h"

#ifndef NO_CPROVER_LIBRARY

extern "C" {
extern uint8_t _binary_clib32_goto_start;
extern uint8_t _binary_clib64_goto_start;
extern uint8_t _binary_clib32_goto_end;
extern uint8_t _binary_clib64_goto_end;

uint8_t *clib_ptrs[2][2] = {
{ &_binary_clib32_goto_start, &_binary_clib32_goto_end},
{ &_binary_clib64_goto_start, &_binary_clib64_goto_end},
};
}

#endif

bool
is_in_list(std::list<irep_idt> &list, irep_idt item)
{

  for(std::list<irep_idt>::const_iterator it = list.begin(); it != list.end();
      it++)
    if (*it == item)
      return true;

  return false;
}

void
generate_symbol_deps(irep_idt name, irept irep, std::multimap<irep_idt, irep_idt> &deps)
{
  std::pair<irep_idt, irep_idt> type;

  forall_irep(irep_it, irep.get_sub()) {
    if (irep_it->id() == "symbol") {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->identifier());
      deps.insert(type);
    } else if (irep_it->id() == "argument") {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->cmt_identifier());
      deps.insert(type);
    } else {
      generate_symbol_deps(name, *irep_it, deps);
    }
  }

  forall_named_irep(irep_it, irep.get_named_sub()) {
    if (irep_it->second.id() == "symbol") {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->second.identifier());
      deps.insert(type);
    } else if (irep_it->second.id() == "argument") {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->second.cmt_identifier());
      deps.insert(type);
    } else {
      generate_symbol_deps(name, irep_it->second, deps);
    }
  }

  return;
}

void
ingest_symbol(irep_idt name, std::multimap<irep_idt, irep_idt> &deps, std::list<irep_idt> &to_include)
{
  std::pair<std::multimap<irep_idt,irep_idt>::const_iterator,
            std::multimap<irep_idt,irep_idt>::const_iterator> range;
  std::multimap<irep_idt, irep_idt>::const_iterator it;

  range = deps.equal_range(name);
  if (range.first == range.second)
    return;

  for (it = range.first; it != range.second; it++)
    to_include.push_back(it->second);

  deps.erase(name);

  return;
}

void add_cprover_library(
  contextt &context,
  message_handlert &message_handler)
{
#ifdef NO_CPROVER_LIBRARY
  return;
#else
  contextt new_ctx, store_ctx;
  goto_functionst goto_functions;
  std::multimap<irep_idt, irep_idt> symbol_deps;
  std::list<irep_idt> to_include;
  ansi_c_languaget ansi_c_language;
  char symname_buffer[288];
  FILE *f;
  uint8_t **this_clib_ptrs;
  unsigned long size;
  int fd;

  if(config.ansi_c.lib==configt::ansi_ct::LIB_NONE)
    return;

  if (config.ansi_c.word_size == 32) {
    this_clib_ptrs = &clib_ptrs[0][0];
  } else if (config.ansi_c.word_size == 64) {
    this_clib_ptrs = &clib_ptrs[1][0];
  } else {
    if (config.ansi_c.word_size == 16) {
      std::cerr << "Warning: this version of ESBMC does not have a C library ";
      std::cerr << "for 16 bit machines";
      return;
    }

    std::cerr << "No c library for bitwidth " << config.ansi_c.int_width << std::endl;
    abort();
  }

  size = (unsigned long)this_clib_ptrs[1] - (unsigned long)this_clib_ptrs[0];
  if (size == 0) {
    std::cerr << "error: Zero-lengthed internal C library" << std::endl;
    abort();
  }

#ifndef _WIN32
  sprintf(symname_buffer, "/tmp/ESBMC_XXXXXX");
  fd = mkstemp(symname_buffer);
  close(fd);
#else
  char tmpdir[256];
  GetTempPath(sizeof(tmpdir), tmpdir);
  GetTempFileName(tmpdir, "bmc", 0, symname_buffer);
#endif
  f = fopen(symname_buffer, "wb");
  if (fwrite(this_clib_ptrs[0], size, 1, f) != 1) {
    std::cerr << "Couldn't manipulate internal C library" << std::endl;
    abort();
  }
  fclose(f);

  std::ifstream infile(symname_buffer, std::ios::in | std::ios::binary);
  read_goto_binary(infile, new_ctx, goto_functions, message_handler);
  unlink(symname_buffer);

  forall_symbols(it, new_ctx.symbols) {
    generate_symbol_deps(it->first, it->second.value, symbol_deps);
    generate_symbol_deps(it->first, it->second.type, symbol_deps);
  }

  /* The code just pulled into store_ctx might use other symbols in the C
   * library. So, repeatedly search for new C library symbols that we use but
   * haven't pulled in, then pull them in. We finish when we've made a pass
   * that adds no new symbols. */

  forall_symbols(it, new_ctx.symbols) {
    symbolst::const_iterator used_sym = context.symbols.find(it->second.name);
    if (used_sym != context.symbols.end() && used_sym->second.value.is_nil()){
      store_ctx.add(it->second);
      ingest_symbol(it->first, symbol_deps, to_include);
    }
  }

  for (std::list<irep_idt>::const_iterator nameit = to_include.begin();
            nameit != to_include.end(); nameit++) {

    symbolst::const_iterator used_sym = new_ctx.symbols.find(*nameit);
    if (used_sym != new_ctx.symbols.end()) {
      store_ctx.add(used_sym->second);
      ingest_symbol(used_sym->first, symbol_deps, to_include);
    }
  }

  ansi_c_language.merge_context(
      context, store_ctx, message_handler, "<built-in-library>");

#endif
}
