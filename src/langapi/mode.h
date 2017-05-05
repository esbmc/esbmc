/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_MODE_H
#define CPROVER_MODE_H

#include <util/language.h>

// Table recording details about language modes

struct mode_table_et
{
  const char *name;
  languaget *(*new_language)();
  const char **extensions;
};

// List of language modes that are going to be supported in the final tool.
// Must be declared by user of langapi, must end with HAVE_MODE_NULL.

extern const mode_table_et mode_table[];

extern const char *extensions_ansi_c[];
extern const char *extensions_cpp[];

#ifndef WITHOUT_CLANG
languaget *new_clang_c_language();
languaget *new_clang_cpp_language();
#endif
languaget *new_ansi_c_language();
languaget *new_cpp_language();

// List of language entries, one can put in the mode table:
#define LANGAPI_HAVE_MODE_CLANG_C \
  { "C",        &new_clang_c_language,   extensions_ansi_c   }
#define LANGAPI_HAVE_MODE_CLANG_CPP \
  { "C",        &new_clang_cpp_language,   extensions_cpp   }
#define LANGAPI_HAVE_MODE_C \
  { "C",        &new_ansi_c_language,   extensions_ansi_c   }
#define LANGAPI_HAVE_MODE_CPP \
  { "C++",      &new_cpp_language,      extensions_cpp      }
#define LANGAPI_HAVE_MODE_END {NULL, NULL, NULL}

int get_mode(const std::string &str);
int get_mode_filename(const std::string &filename);

languaget *new_language(const char *mode);

#define MODE_C        "C"
#define MODE_CPP      "C++"

#endif
