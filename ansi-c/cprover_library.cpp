/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

extern "C" {
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
}

#include <sstream>
#include <istream>
#include <fstream>

#include <config.h>

#include <goto-programs/read_goto_binary.h>

#include "cprover_library.h"
#include "ansi_c_language.h"

/*******************************************************************\

Function: add_cprover_library

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void add_cprover_library(
  contextt &context,
  message_handlert &message_handler)
{
  char symname_buffer[256];
  void *self, *start;
  FILE *f;
  char *filename;
  unsigned long size;

  if(config.ansi_c.lib==configt::ansi_ct::LIB_NONE)
    return;

  self = dlopen(NULL, 0);
  if (self == NULL) {
    std::cerr << "Could not open self linker handle with dlopen" << std::endl;
    abort();
  }

  sprintf(symname_buffer, "_binary_clib%d_goto_start", config.ansi_c.int_width);
  start = dlsym(self, symname_buffer);
  if (start == NULL) {
    std::cerr << "Could not locate internal C library" << std::endl;
    abort();
  }

  sprintf(symname_buffer, "_binary_clib%d_goto_size", config.ansi_c.int_width);
  size = (unsigned long)dlsym(self, symname_buffer);
  if (size == 0) {
    std::cerr << "error: Zero-lengthed internal C library" << std::endl;
    abort();
  }

  dlclose(self);

  sprintf(symname_buffer, "ESBMC_XXXXXX");
  filename = mktemp(symname_buffer);
  f = fopen(filename, "w");
  if (fwrite(start, size, 1, f) != 1) {
    std::cerr << "Couldn't manipulate internal C library" << std::endl;
    abort();
  }
  fclose(f);

  std::ifstream infile(filename);
  read_goto_binary(infile, context, blah, message_handler);

  if(count>0)
  {
    std::istringstream in(library_text.str());
    ansi_c_languaget ansi_c_language;
    ansi_c_language.parse(in, "", message_handler);

    contextt new_context;
    ansi_c_language.typecheck(
      new_context, "<built-in-library>", message_handler);

    ansi_c_language.merge_context(
      context, new_context,
      message_handler, "<built-in-library>");
  }
}

