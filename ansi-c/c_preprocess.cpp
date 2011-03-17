/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdio.h>
#include <stdlib.h>

#ifdef __LINUX__
#include <unistd.h>
#endif

#ifdef __APPLE__
#include <unistd.h>
#endif

#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#endif

#include <fstream>

#include <config.h>
#include <i2string.h>
#include <message_stream.h>

#include "c_preprocess.h"

#define GCC_DEFINES_16 \
  " -D__INT_MAX__=32767"\
  " -D__CHAR_BIT__=8"\
  " -D__WCHAR_MAX__=32767"\
  " -D__SCHAR_MAX__=127"\
  " -D__SHRT_MAX__=32767"\
  " -D__LONG_LONG_MAX__=2147483647L"\
  " -D__LONG_MAX__=2147483647" \
  " -D__FLT_MIN__=1.17549435e-38F" \
  " -D__FLT_MAX__=3.40282347e+38F" \
  " -D__LDBL_MIN__=3.36210314311209350626e-4932L" \
  " -D__LDBL_MAX__=1.18973149535723176502e+4932L" \
  " -D__DBL_MIN__=2.2250738585072014e-308" \
  " -D__DBL_MAX__=1.7976931348623157e+308" \
  " -D __SIZE_TYPE__=\"unsigned int\""\
  " -D __PTRDIFF_TYPE__=int"\
  " -D __WCHAR_TYPE__=int"\
  " -D __WINT_TYPE__=int"\
  " -D __INTMAX_TYPE__=\"long long int\""\
  " -D __UINTMAX_TYPE__=\"long long unsigned int\""

#define GCC_DEFINES_32 \
  " -D__INT_MAX__=2147483647"\
  " -D__CHAR_BIT__=8"\
  " -D__WCHAR_MAX__=2147483647"\
  " -D__SCHAR_MAX__=127"\
  " -D__SHRT_MAX__=32767"\
  " -D__LONG_LONG_MAX__=9223372036854775807LL"\
  " -D__LONG_MAX__=2147483647L" \
  " -D__FLT_MIN__=1.17549435e-38F" \
  " -D__FLT_MAX__=3.40282347e+38F" \
  " -D__LDBL_MIN__=3.36210314311209350626e-4932L" \
  " -D__LDBL_MAX__=1.18973149535723176502e+4932L" \
  " -D__DBL_MIN__=2.2250738585072014e-308" \
  " -D__DBL_MAX__=1.7976931348623157e+308" \
  " -D __SIZE_TYPE__=\"long unsigned int\""\
  " -D __PTRDIFF_TYPE__=int"\
  " -D __WCHAR_TYPE__=int"\
  " -D __WINT_TYPE__=int"\
  " -D __INTMAX_TYPE__=\"long long int\""\
  " -D __UINTMAX_TYPE__=\"long long unsigned int\""
                        
#define GCC_DEFINES_64 \
  " -D__INT_MAX__=2147483647"\
  " -D__CHAR_BIT__=8"\
  " -D__WCHAR_MAX__=2147483647"\
  " -D__SCHAR_MAX__=127"\
  " -D__SHRT_MAX__=32767"\
  " -D__LONG_LONG_MAX__=9223372036854775807LL"\
  " -D__LONG_MAX__=9223372036854775807L"\
  " -D__FLT_MIN__=1.17549435e-38F" \
  " -D__FLT_MAX__=3.40282347e+38F" \
  " -D__LDBL_MIN__=3.36210314311209350626e-4932L" \
  " -D__LDBL_MAX__=1.18973149535723176502e+4932L" \
  " -D__DBL_MIN__=2.2250738585072014e-308" \
  " -D__DBL_MAX__=1.7976931348623157e+308" \
  " -D__x86_64=1"\
  " -D__LP64__=1"\
  " -D__x86_64__=1"\
  " -D_LP64=1"\
  " -D __SIZE_TYPE__=\"long unsigned int\""\
  " -D __PTRDIFF_TYPE__=int"\
  " -D __WCHAR_TYPE__=int"\
  " -D __WINT_TYPE__=int"\
  " -D __INTMAX_TYPE__=\"long long int\""\
  " -D __UINTMAX_TYPE__=\"long long unsigned int\""

/*******************************************************************\

Function: c_preprocess

  Inputs:

 Outputs:

 Purpose: ANSI-C preprocessing

\*******************************************************************/

bool c_preprocess(
  std::istream &instream,
  const std::string &path,
  std::ostream &outstream,
  message_handlert &message_handler)
{
  // preprocessing
  message_streamt message_stream(message_handler);

  std::string file=path,
              stderr_file="tmp.stderr.txt";

  if(path=="") // stdin
  {
    char ch;

    file="tmp.stdin.c";
    FILE *tmp=fopen(file.c_str(), "wt");

    while(instream.read(&ch, 1)!=NULL)
      fputc(ch, tmp);

    fclose(tmp);
  }

  std::string command;
  
  // use VC98 CL in case of WIN32
  
  #ifdef _WIN32
  command="CL /nologo /E /D__ESBMC__";
  command+=" /D__WORDSIZE="+i2string(config.ansi_c.word_size);
  command+=" /D__PTRDIFF_TYPE__=int";
  #else
  command="gcc -E -undef -D__ESBMC__";

  if(config.ansi_c.os!=configt::ansi_ct::OS_WIN32)
  {
    command+=" -D__null=0";
    command+=" -D__WORDSIZE="+i2string(config.ansi_c.word_size);
    command+=" -D__GNUC__=4";
    
    // Tell the system library which standards we support.
    command+=" -D__STRICT_ANSI__=1 -D_POSIX_SOURCE=1 -D_POSIX_C_SOURCE=200112L";

    if(config.ansi_c.word_size==16)
      command+=GCC_DEFINES_16;
    else if(config.ansi_c.word_size==32)
      command+=GCC_DEFINES_32;
    else if(config.ansi_c.word_size==64)
      command+=GCC_DEFINES_64;
  }
    
  if(config.ansi_c.os==configt::ansi_ct::OS_I386_LINUX)
  { // assume we're running i386-linux
    command+=" -Di386 -D__i386 -D__i386__";
    command+=" -Dlinux -D__linux -D__linux__ -D__gnu_linux__";
    command+=" -Dunix -D__unix -D__unix__";
  }
  else if(config.ansi_c.os==configt::ansi_ct::OS_I386_MACOS)
  {
    command+=" -Di386 -D__i386 -D__i386__";
    command+=" -D__APPLE__ -D__MACH__ -D__LITTLE_ENDIAN__";
  }
  else if(config.ansi_c.os==configt::ansi_ct::OS_PPC_MACOS)
  {
    command+=" -D__APPLE__ -D__MACH__ -D__BIG_ENDIAN__";
  }
  else if(config.ansi_c.os==configt::ansi_ct::OS_WIN32)
  {
    command+=" -D _MSC_VER=1400";
    command+=" -D _WIN32";
    command+=" -D _M_IX86=Blend";

    if(config.ansi_c.char_is_unsigned)
      command+=" -D _CHAR_UNSIGNED";
  }
  else
  {
    command+=" -nostdinc"; // make sure we don't mess with the system library
  }
  #endif  
  
  // Standard Defines, ANSI9899 6.10.8
  std::string pre;
  #ifdef _WIN32
    pre = " /D";
  #else
    pre = " -D";
  #endif    
  command += pre + "__STDC_VERSION__=199901L";
  command += pre + "__STDC_IEC_559__=1";
  command += pre + "__STDC_IEC_559_COMPLEX__=1";
  command += pre + "__STDC_ISO_10646__=1";
  
  for(std::list<std::string>::const_iterator
      it=config.ansi_c.defines.begin();
      it!=config.ansi_c.defines.end();
      it++)
    #ifdef _WIN32
    command+=" /D \""+*it+"\"";
    #else
    command+=" -D'"+*it+"'";
    #endif

  for(std::list<std::string>::const_iterator
      it=config.ansi_c.include_paths.begin();
      it!=config.ansi_c.include_paths.end();
      it++)
    #ifdef _WIN32
    command+=" /I \""+*it+"\"";
    #else
    command+=" -I'"+*it+"'";
    #endif

  #ifdef _WIN32
  command+=" \""+file+"\"";
  command+=" 2> \""+stderr_file+"\"";
  #else
  command+=" \""+file+"\"";
  command+=" 2> \""+stderr_file+"\"";
  #endif

  FILE *stream=popen(command.c_str(), "r");

  if(stream!=NULL)
  {
    char ch;
    while((ch=fgetc(stream))!=EOF)
      outstream << ch;

    int result=pclose(stream);
    if(path=="") unlink(file.c_str());

    // errors/warnings
    std::ifstream stderr_stream(stderr_file.c_str());
    while((stderr_stream.read(&ch, 1))!=NULL)
      message_stream.str << ch;

    unlink(stderr_file.c_str());

    if(result!=0)
    {
      message_stream.error_parse(1);
      message_stream.error("Preprocessing failed");
      return true;
    }
    else
      message_stream.error_parse(2);
  }
  else
  {
    if(path=="") unlink(file.c_str());
    unlink(stderr_file.c_str());
    message_stream.error("Preprocessing failed (popen failed)");
    return true;
  }

  return false;
}
