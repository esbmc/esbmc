#ifndef CPROVER_LANGUAGE_FILE_H
#define CPROVER_LANGUAGE_FILE_H

#include <set>
#include <util/context.h>

class language_modulet
{
public:
  std::string name;
  bool type_checked, in_progress;
  class language_filet *file;

  language_modulet()
  {
    type_checked = in_progress = false;
  }
};

class language_filet
{
public:
  std::set<std::string> modules;

  class languaget *language;
  std::string filename;

  void get_modules();

  language_filet()
  {
    language = nullptr;
  }
  ~language_filet();
};

class language_filest
{
public:
  typedef std::map<std::string, language_filet> filemapt;
  filemapt filemap;

  typedef std::map<std::string, language_modulet> modulemapt;
  modulemapt modulemap;

  void clear_files()
  {
    filemap.clear();
  }

  bool parse();

  void show_parse(std::ostream &out);

  bool typecheck(contextt &context);

  bool final(contextt &context);

  bool interfaces(contextt &context);

  void clear()
  {
    filemap.clear();
    modulemap.clear();
  }

protected:
  bool typecheck_module(contextt &context, language_modulet &module);

  bool typecheck_module(contextt &context, const std::string &module);

  void typecheck_virtual_methods(contextt &context);
};

#endif
