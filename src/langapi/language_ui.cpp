/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <fstream>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <memory>
#include <util/i2string.h>
#include <util/show_symbol_table.h>

language_uit::language_uit(const cmdlinet &__cmdline, const messaget &msg)
  : language_files(msg), context(msg), _cmdline(__cmdline), msg(msg)
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
  int mode = get_mode_filename(filename);

  if(mode < 0)
  {
    msg.error("failed to figure out type of file", filename);
    return true;
  }

  if(config.options.get_bool_option("old-frontend"))
    mode++;

  // Check that it opens
  std::ifstream infile(filename.c_str());
  if(!infile)
  {
    msg.error("failed to open input file", filename);
    return true;
  }

  language_filet language_file;

  std::pair<language_filest::filemapt::iterator, bool> result =
    language_files.filemap.insert(
      std::pair<std::string, language_filet>(filename, language_file));

  language_filet &lf = result.first->second;
  lf.filename = filename;
  lf.language = mode_table[mode].new_language();
  languaget &language = *lf.language;

  msg.status("Parsing", filename);

  if(language.parse(filename, msg))
  {
    msg.error("PARSING ERROR");
    return true;
  }

  lf.get_modules();

  return false;
}

bool language_uit::typecheck()
{
  msg.status("Converting");

  if(language_files.typecheck(context))
  {
    msg.error("CONVERSION ERROR");
    return true;
  }

  return false;
}

bool language_uit::final()
{
  if(language_files.final(context))
  {
    msg.error("CONVERSION ERROR");
    return true;
  }

  return false;
}

void language_uit::show_symbol_table()
{
}

void language_uit::show_symbol_table_xml_ui()
{
  msg.error("cannot show symbol table in this format");
}

void language_uit::show_symbol_table_plain(std::ostream &out)
{
  ::show_symbol_table_plain(namespacet(context), out);
}
