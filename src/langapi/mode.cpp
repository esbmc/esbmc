#include <cassert>
#include <cstring>
#include <langapi/mode.h>
#include <util/config.h>
#include <util/language.h>

namespace
{
struct language_desct
{
  const char *name;
  const char *const *filename_extensions;
};
} // namespace

static const char *const extensions_ansi_c[] = {"c", "i", nullptr};

#ifdef _WIN32
static const char *const extensions_cpp[] =
  {"cpp", "cc", "cu", "ipp", "cxx", NULL};
#else
static const char *const extensions_cpp[] =
  {"cpp", "cc", "cu", "ipp", "C", "cxx", nullptr};
#endif

static const char *const extensions_sol_ast[] = {"solast", nullptr};
static const char *const extensions_jimple[] = {"jimple", nullptr};
static const char *const extensions_python[] = {"py", nullptr};

static const language_desct language_desc_C = {"C", extensions_ansi_c};
static const language_desct language_desc_CPP = {"C++", extensions_cpp};
static const language_desct language_desc_Solidity = {
  "Solidity",
  extensions_sol_ast};
static const language_desct language_desc_Jimple = {
  "Jimple",
  extensions_jimple};

static const language_desct language_desc_python = {
  "Python",
  extensions_python};

static const language_desct *language_desc(language_idt id)
{
  switch (id)
  {
  case language_idt::NONE:
    break;
  case language_idt::C:
    return &language_desc_C;
  case language_idt::CPP:
    return &language_desc_CPP;
  case language_idt::SOLIDITY:
    return &language_desc_Solidity;
  case language_idt::JIMPLE:
    return &language_desc_Jimple;
  case language_idt::PYTHON:
    return &language_desc_python;
  }
  return nullptr;
}

const char *language_name(language_idt lang)
{
  const language_desct *desc = language_desc(lang);
  return desc ? desc->name : nullptr;
}

language_idt language_id_by_name(const std::string &name)
{
  for (int i = 0;; i++)
  {
    language_idt lid = static_cast<language_idt>(i);
    const language_desct *desc = language_desc(lid);
    if (!desc)
      return language_idt::NONE;
    if (desc->name == name)
      return lid;
  }
}

language_idt language_id_by_path(const std::string &path)
{
  const char *dot = strrchr(path.c_str(), '.');

  if (dot == nullptr)
    return language_idt::NONE;

  std::string_view ext(dot + 1);

  for (int i = 0;; i++)
  {
    language_idt lid = static_cast<language_idt>(i);
    const language_desct *desc = language_desc(lid);
    if (!desc)
      return language_idt::NONE;
    for (const char *const *e = desc->filename_extensions; *e; e++)
      if (*e == ext)
        return lid;
  }
}

static int get_mode(language_idt lang)
{
  assert(language_desc(lang));

  for (int i = 0; mode_table[i].new_language; i++)
    if (lang == mode_table[i].language_id)
      return i;

  return -1;
}

static int get_old_frontend_mode(int current_mode)
{
  language_idt expected = mode_table[current_mode].language_id;
  for (int i = current_mode + 1; mode_table[i].new_language; i++)
    if (expected == mode_table[i].language_id)
      return i;

  return -1;
}

std::unique_ptr<languaget> new_language(language_idt lang)
{
  int mode = get_mode(lang);

  if (mode >= 0 && config.options.get_bool_option("old-frontend"))
    mode = get_old_frontend_mode(mode);

  languaget *l = nullptr;
  if (mode >= 0)
    l = mode_table[mode].new_language();

  return std::unique_ptr<languaget>(l);
}
