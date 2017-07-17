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

static ui_message_handlert::uit get_ui_cmdline(const cmdlinet &cmdline)
{
  if(cmdline.isset("gui"))
    return ui_message_handlert::OLD_GUI;
  else if(cmdline.isset("xml-ui"))
    return ui_message_handlert::XML_UI;

  return ui_message_handlert::PLAIN;
}

language_uit::language_uit(const cmdlinet &__cmdline):
  ui_message_handler(get_ui_cmdline(__cmdline)),
  _cmdline(__cmdline)
{
  set_message_handler(&ui_message_handler);
}

bool language_uit::parse()
{
  for(const auto & arg : _cmdline.args)
  {
    if(parse(arg))
      return true;
  }

  return false;
}

bool language_uit::parse(const std::string &filename)
{
  int mode=get_mode_filename(filename);

  if(mode<0)
  {
    error("failed to figure out type of file", filename);
    return true;
  }

  if(config.options.get_bool_option("old-frontend"))
  {
#ifndef WITHOUT_CLANG
    mode++;
#else
    std::cerr << "The clang frontend has not been built into this version of ESBMC, sorry" << std::endl;
    abort();
#endif
  }

  // Check that it opens
  std::ifstream infile(filename.c_str());
  if(!infile)
  {
    error("failed to open input file", filename);
    return true;
  }

  language_filet language_file;

  std::pair<language_filest::filemapt::iterator, bool>
    result=language_files.filemap.insert(
      std::pair<std::string, language_filet>(filename, language_file));

  language_filet &lf=result.first->second;
  lf.filename=filename;
  lf.language=mode_table[mode].new_language();
  languaget &language=*lf.language;

  status("Parsing", filename);

  if(language.parse(filename, *get_message_handler()))
  {
    if(get_ui()==ui_message_handlert::PLAIN)
      std::cerr << "PARSING ERROR" << std::endl;

    return true;
  }

  lf.get_modules();

  return false;
}

bool language_uit::typecheck()
{
  status("Converting");

  language_files.set_message_handler(message_handler);
  language_files.set_verbosity(get_verbosity());

  if(language_files.typecheck(context))
  {
    if(get_ui()==ui_message_handlert::PLAIN)
      std::cerr << "CONVERSION ERROR" << std::endl;

    return true;
  }

  return false;
}

bool language_uit::final()
{
  language_files.set_message_handler(message_handler);
  language_files.set_verbosity(get_verbosity());

  if(language_files.final(context))
  {
    if(get_ui()==ui_message_handlert::PLAIN)
      std::cerr << "CONVERSION ERROR" << std::endl;

    return true;
  }

  return false;
}

void language_uit::show_symbol_table()
{
  switch(get_ui())
  {
  case ui_message_handlert::PLAIN:
    show_symbol_table_plain(std::cout);
    break;

  case ui_message_handlert::XML_UI:
    show_symbol_table_xml_ui();
    break;

  default:
    error("cannot show symbol table in this format");
  }
}

void language_uit::show_symbol_table_xml_ui()
{
  error("cannot show symbol table in this format");
}

void language_uit::show_symbol_table_plain(std::ostream &out)
{
  ::show_symbol_table_plain(namespacet(context), out);
}
