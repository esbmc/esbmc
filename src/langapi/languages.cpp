/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <langapi/languages.h>
#include <langapi/mode.h>

languagest::languagest(
  const namespacet &_ns,
  language_idt lang,
  const messaget &msg)
  : ns(_ns), msg(msg)
{
  language = new_language(lang, msg);
}

languagest::~languagest()
{
  delete language;
}
