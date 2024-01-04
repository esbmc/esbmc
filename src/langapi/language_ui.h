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
  namespacet ns;

  language_uit();
  virtual ~language_uit() noexcept = default;

  /* The instance of this class manages the global migrate_namespace_lookup,
   * thus it cannot be copied. */
  language_uit(language_uit &&) noexcept;
  language_uit &operator=(language_uit &&) noexcept;

  virtual bool parse(const cmdlinet &cmdline);
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
};

#endif
