/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

extern "C" {
#include <stdio.h>
#include <stdlib.h>
}

#include <sstream>
#include <istream>
#include <fstream>

#include <config.h>

#include <goto-programs/read_goto_binary.h>

#include "cprover_library.h"
#include "ansi_c_language.h"

extern void *_binary_clib16_goto_size;
extern void *_binary_clib32_goto_size;
extern void *_binary_clib64_goto_size;
extern void *_binary_clib16_goto_start;
extern void *_binary_clib32_goto_start;
extern void *_binary_clib64_goto_start;
extern void *_binary_clib16_goto_end;
extern void *_binary_clib32_goto_end;
extern void *_binary_clib64_goto_end;

void *clib_ptrs[3][3] = {
{ _binary_clib16_goto_size, _binary_clib16_goto_start, _binary_clib16_goto_end},
{ _binary_clib32_goto_size, _binary_clib32_goto_start, _binary_clib32_goto_end},
{ _binary_clib64_goto_size, _binary_clib64_goto_start, _binary_clib64_goto_end},
};

void add_cprover_library(
  contextt &context,
  message_handlert &message_handler)
{
  contextt new_ctx;
  goto_functionst goto_functions;
  ansi_c_languaget ansi_c_language;
  char symname_buffer[256];
  FILE *f;
  void **this_clib_ptrs;
  unsigned long size;
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

  size = (unsigned long)this_clib_ptrs[0];
  if (size == 0) {
    std::cerr << "error: Zero-lengthed internal C library" << std::endl;
    abort();
  }

  sprintf(symname_buffer, "ESBMC_XXXXXX");
  fd = mkstemp(symname_buffer);
  close(fd);
  f = fopen(symname_buffer, "w");
  if (fwrite(this_clib_ptrs[1], size, 1, f) != 1) {
    std::cerr << "Couldn't manipulate internal C library" << std::endl;
    abort();
  }
  fclose(f);

  std::ifstream infile(symname_buffer);
  read_goto_binary(infile, new_ctx, goto_functions, message_handler);

  ansi_c_language.merge_context(
        context, new_ctx, message_handler, "<built-in-library>");
}
