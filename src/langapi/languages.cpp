/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <langapi/languages.h>
#include <langapi/mode.h>

/*******************************************************************\

Function: languagest::languagest

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languagest::languagest(const namespacet &_ns, const char* mode):ns(_ns)
{
  language=new_language(mode);
}

/*******************************************************************\

Function: languagest::~languagest

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languagest::~languagest()
{
  delete language;
}

