#ifndef STL_DEFINITIONS
#define STL_DEFINITIONS

#include "cstddef"

#define SIGINT 2

#define SEEK_SET 0
#define SEEK_CUR 1
#define SEEK_END 2

#ifndef __TIMESTAMP__
#  define __TIMESTAMP__ (0)
#endif

#ifdef _WIN64
typedef __int64 streamsize;
#else
typedef unsigned int streamsize;
#endif

/* The <iomanip> manipulators are implementation-defined types; the model
 * funnels all of them through this one, tagged with which setting to apply
 * when it reaches a stream (github #7016). */
class smanip
{
public:
  enum kind
  {
    _setiosflags,
    _resetiosflags,
    _setbase,
    _setfill,
    _setprecision,
    _setw
  };

  int _kind;
  long _arg;

  smanip(kind k, long a) : _kind(k), _arg(a)
  {
  }
};

#define _SIZE_T_DEFINED

#endif
