/*******************************************************************\

Module: 

Author: CM Wintersteiger

\*******************************************************************/

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif

#include <stdlib.h>

#ifdef __MACH__
#include <unistd.h>
#endif

#ifdef __linux__
#include <unistd.h>
#endif

#include "tempdir.h"

bool get_temporary_directory(char *t)
{
  #ifdef _WIN32    
    DWORD dwBufSize = MAX_PATH;
    char lpPathBuffer[MAX_PATH];
    DWORD dwRetVal = GetTempPath(dwBufSize, lpPathBuffer);
    if (dwRetVal > dwBufSize || (dwRetVal == 0))
      return true;
    UINT uRetVal = GetTempFileName(lpPathBuffer, "TLO", 0, t);
    if (uRetVal == 0)
      return true;
    unlink(t);
    if (_mkdir(t)!=0)
      return true;
  #else    
    char *td = mkdtemp(t);
    if(!td) return true;
  #endif
    
  return false;
}
