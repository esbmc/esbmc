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
  languaget *(*new_language)(const messaget &msg);
  const char **extensions;
};

// List of language modes that are going to be supported in the final tool.
// Must be declared by user of langapi, must end with HAVE_MODE_NULL.

extern const mode_table_et mode_table[];

extern const char *extensions_ansi_c[];
extern const char *extensions_cpp[];
extern const char *extensions_sol_ast[];
extern const char *extensions_jimple[];

languaget *new_clang_c_language(const messaget &msg);
languaget *new_clang_cpp_language(const messaget &msg);
languaget *new_jimple_language(const messaget &msg);
languaget *new_ansi_c_language(const messaget &msg);
languaget *new_cpp_language(const messaget &msg);
languaget *new_solidity_language(const messaget &msg);

// List of language entries, one can put in the mode table:
#define LANGAPI_HAVE_MODE_CLANG_C                                              \
  {                                                                            \
    "C", &new_clang_c_language, extensions_ansi_c                              \
  }
#define LANGAPI_HAVE_MODE_CLANG_CPP                                            \
  {                                                                            \
    "C", &new_clang_cpp_language, extensions_cpp                               \
  }
#define LANGAPI_HAVE_MODE_SOLAST                                               \
  {                                                                            \
    "Solidity AST", &new_solidity_language, extensions_sol_ast                 \
  }
#define LANGAPI_HAVE_MODE_C                                                    \
  {                                                                            \
    "C", &new_ansi_c_language, extensions_ansi_c                               \
  }
#define LANGAPI_HAVE_MODE_CPP                                                  \
  {                                                                            \
    "C++", &new_cpp_language, extensions_cpp                                   \
  }
#define LANGAPI_HAVE_MODE_JIMPLE                                               \
  {                                                                            \
    "Jimple", &new_jimple_language, extensions_jimple                          \
  }

#define LANGAPI_HAVE_MODE_END                                                  \
  {                                                                            \
    NULL, NULL, NULL                                                           \
  }

int get_mode(const std::string &str);
int get_mode_filename(const std::string &filename);
int get_old_frontend_mode(int current_mode);

languaget *new_language(const char *mode, const messaget &msg);

#define MODE_C "C"
#define MODE_CPP "C++"

#endif
