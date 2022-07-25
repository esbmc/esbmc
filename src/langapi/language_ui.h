#ifndef CPROVER_LANGUAGE_UI_H
#define CPROVER_LANGUAGE_UI_H

#include <util/language.h>
#include <util/language_file.h>
#include <util/parseoptions.h>

class language_uit
{
public:
  language_filest language_files;
  contextt context;

  language_uit(const cmdlinet &__cmdline);
  virtual ~language_uit() = default;

  virtual bool parse();
  virtual bool parse(const std::string &filename);
  virtual bool typecheck();
  virtual bool final();

  virtual void clear_parse()
  {
    language_files.clear();
  }

  virtual void show_symbol_table();
  virtual void show_symbol_table_plain(std::ostream &out);
  virtual void show_symbol_table_xml_ui();

protected:
  const cmdlinet &_cmdline;
};

#endif
