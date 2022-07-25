#ifndef CPROVER_MODE_H
#define CPROVER_MODE_H

#include <string>

/* forward declarations */
class languaget;

enum class language_idt : int
{
  NONE = -1,
  C,
  CPP,
  SOLIDITY,
  JIMPLE
};

struct language_desct
{
  const char *name;
  const char *const *filename_extensions;
};

const language_desct *language_desc(language_idt id);
language_idt language_id_by_name(const std::string &name);
language_idt language_id_by_ext(const std::string &ext);
language_idt language_id_by_path(const std::string &path);

// Table recording details about language modes

struct mode_table_et
{
  language_idt language_id;
  languaget *(*new_language)();
};

// List of language modes that are going to be supported in the final tool.
// Must be declared by user of langapi, must end with HAVE_MODE_NULL.

extern const mode_table_et mode_table[];

languaget *new_clang_c_language();
languaget *new_clang_cpp_language();
languaget *new_jimple_language();
languaget *new_ansi_c_language();
languaget *new_cpp_language();
languaget *new_solidity_language();

// List of language entries, one can put in the mode table:
#define LANGAPI_MODE_CLANG_C                                                   \
  {                                                                            \
    language_idt::C, &new_clang_c_language                                     \
  }
#define LANGAPI_MODE_CLANG_CPP                                                 \
  {                                                                            \
    language_idt::CPP, &new_clang_cpp_language                                 \
  }
#define LANGAPI_MODE_SOLAST                                                    \
  {                                                                            \
    language_idt::SOLIDITY, &new_solidity_language                             \
  }
#define LANGAPI_MODE_C                                                         \
  {                                                                            \
    language_idt::C, &new_ansi_c_language                                      \
  }
#define LANGAPI_MODE_CPP                                                       \
  {                                                                            \
    language_idt::CPP, &new_cpp_language                                       \
  }
#define LANGAPI_MODE_JIMPLE                                                    \
  {                                                                            \
    language_idt::JIMPLE, &new_jimple_language                                 \
  }

#define LANGAPI_MODE_END                                                       \
  {                                                                            \
    language_idt::NONE, NULL                                                   \
  }

int get_mode(language_idt lang);
int get_mode(const std::string &str);
int get_mode_filename(const std::string &filename);
int get_old_frontend_mode(int current_mode);

languaget *new_language(language_idt lang);

#endif
