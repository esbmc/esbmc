#include <fstream>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <memory>
#include <util/i2string.h>
#include <util/message.h>
#include <util/show_symbol_table.h>

language_uit::language_uit() : ns(context)
{
  // Ahem
  migrate_namespace_lookup = &ns;
}

language_uit::language_uit(language_uit &&o) noexcept
  : language_files(std::move(o.language_files)),
    context(std::move(o.context)),
    ns(context)
{
  // Ahem
  migrate_namespace_lookup = &ns;
}

language_uit &language_uit::operator=(language_uit &&o) noexcept
{
  language_files = std::move(o.language_files);
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
  int mode = get_mode(lang);

  if (mode < 0)
  {
    log_error("failed to figure out type of file {}", filename);
    return true;
  }

  if (config.options.get_bool_option("old-frontend"))
  {
    mode = get_old_frontend_mode(mode);
    if (mode == -1)
    {
      log_error("old-frontend was not built on this version of ESBMC");
      return true;
    }
  }

  config.language = lang;

  // Check that it opens
  std::ifstream infile(filename.c_str());
  if (!infile)
  {
    log_error("failed to open input file {}", filename);
    return true;
  }

  std::pair<language_filest::filemapt::iterator, bool> result =
    language_files.filemap.emplace(
      std::piecewise_construct,
      std::forward_as_tuple(filename),
      std::tuple<>{});
  assert(result.second);

  language_filet &lf = result.first->second;
  lf.filename = filename;
  lf.language = mode_table[mode].new_language();
  languaget &language = *lf.language;

  log_progress("Parsing {}", filename);

#ifdef ENABLE_SOLIDITY_FRONTEND
  if (mode == get_mode(language_idt::SOLIDITY))
  {
    std::string fun = config.options.get_option("function");
    if (!fun.empty())
      language.set_func_name(fun);

    if (config.options.get_option("sol") == "")
    {
      log_error("Please set the smart contract source file.");
      return true;
    }
    else
    {
      language.set_smart_contract_source(config.options.get_option("sol"));
    }
  }
#endif

  if (language.parse(filename))
  {
    log_error("PARSING ERROR");
    return true;
  }

  lf.get_modules();

  return false;
}

bool language_uit::typecheck()
{
  log_progress("Converting");

  if (language_files.typecheck(context))
  {
    log_error("CONVERSION ERROR");
    return true;
  }

  return false;
}

bool language_uit::final()
{
  if (language_files.final(context))
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
