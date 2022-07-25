#include <fstream>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <memory>
#include <util/i2string.h>
#include <util/message.h>
#include <util/show_symbol_table.h>

language_uit::language_uit(const cmdlinet &__cmdline) : _cmdline(__cmdline)
{
}

bool language_uit::parse()
{
  for(const auto &arg : _cmdline.args)
  {
    if(parse(arg))
      return true;
  }

  return false;
}

bool language_uit::parse(const std::string &filename)
{
  language_idt lang = language_id_by_path(filename);
  int mode = get_mode(lang);

  if(mode < 0)
  {
    log_error("failed to figure out type of file", filename);
    return true;
  }

  if(config.options.get_bool_option("old-frontend"))
  {
    mode = get_old_frontend_mode(mode);
    if(mode == -1)
    {
      log_error("old-frontend was not built on this version of ESBMC");
      return true;
    }
  }

  config.language = lang;

  // Check that it opens
  std::ifstream infile(filename.c_str());
  if(!infile)
  {
    log_error("failed to open input file", filename);
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

  log_status("Parsing", filename);

#ifdef ENABLE_SOLIDITY_FRONTEND
  if(mode == get_mode(language_idt::SOLIDITY))
  {
    language.set_func_name(_cmdline.vm["function"].as<std::string>());

    if(config.options.get_option("contract") == "")
    {
      log_error("Please set the smart contract source file.");
      return true;
    }
    else
    {
      language.set_smart_contract_source(config.options.get_option("contract"));
    }
  }
#endif

  if(language.parse(filename))
  {
    log_error("PARSING ERROR");
    return true;
  }

  lf.get_modules();

  return false;
}

bool language_uit::typecheck()
{
  log_status("Converting");

  if(language_files.typecheck(context))
  {
    log_error("CONVERSION ERROR");
    return true;
  }

  return false;
}

bool language_uit::final()
{
  if(language_files.final(context))
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
  ::show_symbol_table_plain(namespacet(context), out);
}
