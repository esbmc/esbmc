/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <fstream>
#include <langapi/language_ui.h>
#include <langapi/mode.h>
#include <memory>
#include <util/i2string.h>

/*******************************************************************\

Function: language_uit::language_uit

  Inputs:

 Outputs:

 Purpose: Constructor

\*******************************************************************/

static ui_message_handlert::uit get_ui_cmdline(const cmdlinet &cmdline)
{
  if(cmdline.isset("gui"))
    return ui_message_handlert::OLD_GUI;
  else if(cmdline.isset("xml-ui"))
    return ui_message_handlert::XML_UI;

  return ui_message_handlert::PLAIN;
}

/*******************************************************************\

Function: language_uit::language_uit

  Inputs:

 Outputs:

 Purpose: Constructor

\*******************************************************************/

language_uit::language_uit(const cmdlinet &__cmdline):
  ui_message_handler(get_ui_cmdline(__cmdline)),
  _cmdline(__cmdline)
{
  set_message_handler(&ui_message_handler);
}

/*******************************************************************\

Function: language_uit::~language_uit

  Inputs:

 Outputs:

 Purpose: Destructor

\*******************************************************************/

language_uit::~language_uit()
{
}

/*******************************************************************\

Function: language_uit::parse()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool language_uit::parse()
{
  for(unsigned i=0; i<_cmdline.args.size(); i++)
  {
    if(parse(_cmdline.args[i]))
      return true;
  }

  return false;
}

/*******************************************************************\

Function: language_uit::parse()

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool language_uit::parse(const std::string &filename)
{
  int mode=get_mode_filename(filename);

  if(mode<0)
  {
    error("failed to figure out type of file", filename);
    return true;
  }

  if(config.options.get_bool_option("clang-frontend"))
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

/*******************************************************************\

Function: language_uit::typecheck

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: language_uit::final

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: language_uit::show_symbol_table

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: language_uit::show_symbol_table_xml_ui

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void language_uit::show_symbol_table_xml_ui()
{
  error("cannot show symbol table in this format");
}

/*******************************************************************\

Function: language_uit::show_symbol_table_plain

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void language_uit::show_symbol_table_plain(std::ostream &out)
{
  out << std::endl << "Symbols:" << std::endl;
  out << "Number of symbols: " << context.size() << std::endl;
  out << std::endl;

  const namespacet ns(context);
  context.foreach_operand_in_order(
    [&out, &ns] (const symbolt& s)
    {
      int mode;

      if(s.mode=="")
        mode=0;
      else
      {
        mode=get_mode(id2string(s.mode));
        if(mode<0) throw "symbol "+id2string(s.name)+" has unknown mode";
      }

      std::unique_ptr<languaget> p(mode_table[mode].new_language());
      std::string type_str, value_str;

      if(s.type.is_not_nil())
        p->from_type(s.type, type_str, ns);

      if(s.value.is_not_nil())
        p->from_expr(s.value, value_str, ns);

      out << "Symbol......: " << s.name << std::endl;
      out << "Pretty name.: " << s.pretty_name << std::endl;
      out << "Module......: " << s.module << std::endl;
      out << "Base name...: " << s.base_name << std::endl;
      out << "Mode........: " << s.mode << " (" << mode << ")" << std::endl;
      out << "Type........: " << type_str << std::endl;
      out << "Value.......: " << value_str << std::endl;
      out << "Flags.......:";

      if(s.lvalue)          out << " lvalue";
      if(s.static_lifetime) out << " static_lifetime";
      if(s.file_local)      out << " file_local";
      if(s.is_type)         out << " type";
      if(s.is_extern)       out << " extern";
      if(s.is_macro)        out << " macro";
      if(s.is_used)         out << " used";

      out << std::endl;
      out << "Location....: " << s.location << std::endl;

      out << std::endl;
    }
  );
}
