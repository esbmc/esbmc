#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <memory>
#include <util/base/filesystem.h>
#include <util/base/i2string.h>
#include <util/message/message.h>
#include <util/symtab/show_symbol_table.h>

language_uit::language_uit() : ns(context)
{
  // Ahem
  migrate_namespace_lookup = &ns;
}

language_uit::language_uit(language_uit &&o) noexcept
  : context(std::move(o.context)), ns(context)
{
  // Ahem
  migrate_namespace_lookup = &ns;
}

language_uit &language_uit::operator=(language_uit &&o) noexcept
{
  context = std::move(o.context);
  ns = namespacet(context);

  // Ahem
  migrate_namespace_lookup = &ns;

  return *this;
}

bool language_uit::parse(const cmdlinet &cmdline)
{
  for (const auto &arg : cmdline.args)
  {
    if (parse(arg))
      return true;
  }

  return false;
}

bool language_uit::parse(const std::string &filename)
{
  language_idt lang = language_id_by_path(filename);
  if (lang == language_idt::NONE)
  {
    log_error("failed to figure out type of file {}", filename);
    return true;
  }

  config.language.lid = lang;

  // c2goto names the models it compiles by VFS path, and those exist only in
  // .rodata.
  if (!file_operations::filesystemt::get().readable(filename))
  {
    log_error("failed to open input file {}", filename);
    return true;
  }

  log_progress("Parsing {}", filename);

  auto it = langmap.find(lang);
  if (it == langmap.end())
  {
    auto emplace = langmap.emplace(lang, new_language(lang));
    assert(emplace.second);
    it = emplace.first;
  }

  if (!it->second)
  {
    log_error(
      "{}frontend for {} was not built on this version of ESBMC",
      config.options.get_bool_option("old-frontend") ? "old-" : "",
      language_name(lang));
    return true;
  }

  if (it->second->parse(filename))
  {
    log_error("PARSING ERROR");
    return true;
  }

  return false;
}

bool language_uit::typecheck()
{
  log_progress("Converting");

  for (auto &it : langmap)
    if (it.second->typecheck(context, ""))
    {
      log_error("CONVERSION ERROR");
      return true;
    }

  return false;
}

bool language_uit::final()
{
  for (auto &it : langmap)
    if (it.second->final(context))
    {
      log_error("CONVERSION ERROR");
      return true;
    }

  return false;
}

void language_uit::show_symbol_table()
{
}

void language_uit::show_symbol_table_xml_ui()
{
  log_error("cannot show symbol table in this format");
}

void language_uit::show_symbol_table_plain(std::ostream &out)
{
  ::show_symbol_table_plain(ns, out);
}
