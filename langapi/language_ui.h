/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_LANGUAGE_UI_H
#define CPROVER_LANGUAGE_UI_H

#include <message.h>
#include <util/parseoptions.h>
#include <language_file.h>
#include <language.h>
#include <ui_message.h>

class language_uit:public messaget
{
public:
  language_filest language_files;
  contextt context;

  language_uit(const cmdlinet &__cmdline);
  virtual ~language_uit();

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

  typedef ui_message_handlert::uit uit;

  uit get_ui()
  {
    return ui_message_handler.get_ui();
  }

  ui_message_handlert ui_message_handler;

protected:
  const cmdlinet &_cmdline;

  // k-induction related
  u_int k_step;
  bool base_case;
  bool forward_condition;

  contextt context_base_case_forward_condition;
  contextt context_inductive_step;
};

#endif
