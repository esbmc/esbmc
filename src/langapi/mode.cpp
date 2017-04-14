/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <cstring>
#include <langapi/mode.h>

const char *extensions_ansi_c  []={ "c", "i", NULL };

#ifdef _WIN32
const char *extensions_cpp     []={ "cpp", "cc", "ipp", "cxx", NULL };
#else
const char *extensions_cpp     []={ "cpp", "cc", "ipp", "C", "cxx", NULL };
#endif

/*******************************************************************\

Function: get_mode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_mode(const std::string &str)
{
  unsigned i;

  for(i=0; mode_table[i].name!=NULL; i++)
    if(str==mode_table[i].name)
      return i;

  return -1;
}

/*******************************************************************\

Function: get_mode_filename

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_mode_filename(const std::string &filename)
{
  const char *ext=strrchr(filename.c_str(), '.');

  if(ext==NULL) return -1;

  std::string extension=ext+1;

  if(extension=="") return -1;

  int mode;
  for(mode=0; mode_table[mode].name!=NULL; mode++)
    for(unsigned i=0;
        mode_table[mode].extensions[i]!=NULL;
        i++)
      if(mode_table[mode].extensions[i]==extension)
        return mode;

  return -1;
}

/*******************************************************************\

Function: new_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languaget *new_language(const char *mode)
{
  return (*mode_table[get_mode(mode)].new_language)();
}

