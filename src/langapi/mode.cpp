#include <cassert>
#include <cstring>
#include <langapi/mode.h>

static const char *const extensions_ansi_c[] = {"c", "i", nullptr};

#ifdef _WIN32
static const char *const extensions_cpp[] =
  {"cpp", "cc", "cu", "ipp", "cxx", NULL};
#else
static const char *const extensions_cpp[] =
  {"cpp", "cc", "cu", "ipp", "C", "cxx", nullptr};
#endif

static const char *const extensions_sol_ast[] = {"solast", nullptr};
static const char *extensions_jimple[] = {"jimple", nullptr};
static const language_desct language_desc_C = {"C", extensions_ansi_c};
static const language_desct language_desc_CPP = {"C++", extensions_cpp};
static const language_desct language_desc_Solidity = {
  "Solidity",
  extensions_sol_ast};
static const language_desct language_desc_Jimple = {
  "Jimple",
  extensions_jimple};

const struct language_desct *language_desc(language_idt id)
{
  switch(id)
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
  }
  return nullptr;
}

language_idt language_id_by_name(const std::string &name)
{
  for(int i = 0;; i++)
  {
    language_idt lid = static_cast<language_idt>(i);
    const language_desct *desc = language_desc(lid);
    if(!desc)
      return language_idt::NONE;
    if(desc->name == name)
      return lid;
  }
}

language_idt language_id_by_ext(const std::string &ext)
{
  for(int i = 0;; i++)
  {
    language_idt lid = static_cast<language_idt>(i);
    const language_desct *desc = language_desc(lid);
    if(!desc)
      return language_idt::NONE;
    for(const char *const *e = desc->filename_extensions; *e; e++)
      if(*e == ext)
        return lid;
  }
}

language_idt language_id_by_path(const std::string &path)
{
  const char *ext = strrchr(path.c_str(), '.');

  if(ext == nullptr)
    return language_idt::NONE;

  std::string extension = ext + 1;

  if(extension == "")
    return language_idt::NONE;

  return language_id_by_ext(extension);
}

int get_mode(language_idt lang)
{
  assert(language_desc(lang));

  for(int i = 0; mode_table[i].new_language; i++)
    if(lang == mode_table[i].language_id)
      return i;

  return -1;
}

int get_mode(const std::string &str)
{
  language_idt id = language_id_by_name(str);
  if(id == language_idt::NONE)
    return -1;

  return get_mode(id);
}

int get_old_frontend_mode(int current_mode)
{
  language_idt expected = mode_table[current_mode].language_id;
  for(int i = current_mode + 1; mode_table[i].new_language; i++)
    if(expected == mode_table[i].language_id)
      return i;

  return -1;
}

int get_mode_filename(const std::string &filename)
{
  language_idt id = language_id_by_path(filename);
  if(id == language_idt::NONE)
    return -1;

  return get_mode(id);
}

languaget *new_language(language_idt lang)
{
  return mode_table[get_mode(lang)].new_language();
}
