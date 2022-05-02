/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <langapi/languages.h>
#include <langapi/mode.h>

languagest::languagest(
  const namespacet &_ns,
  const char *mode,
  const messaget &msg)
  : ns(_ns), msg(msg)
{
  language = new_language(mode, msg);
}

languagest::~languagest()
{
  delete language;
}
