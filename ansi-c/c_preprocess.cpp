/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <fstream>

#include <config.h>
#include <i2string.h>
#include <message_stream.h>

#include "c_preprocess.h"

extern "C" {
#include "cpp/iface.h"
}

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

#define STRABS_DEFS \
  " -Dgetopt=getopt_strabs " \
  " -Dfopen=fopen_strabs " \
  " -Dfgets=fgets_strabs " \
  " -Dfputs=fputs_strabs " \
  " -Dcalloc=calloc_strabs " \
  " -Datoi=atoi_strabs " \
  " -Datol=atol_strabs " \
  " -Dgetenv=getenv_strabs " \
  " -Dstrcpy=strcpy_strabs " \
  " -Dstrncpy=strncpy_strabs " \
  " -Dstrcat=strcat_strabs " \
  " -Dstrncat=strncat_strnabs " \
  " -Dstrcmp=strcmp_strabs " \
  " -Dstrncmp=strncmp_strabs " \
  " -Dstrlen=strlen_strabs " \
  " -Dstrdup=strdup_strabs " \
  " -Dmemcpy=memcpy_strabs " \
  " -Dmemset=memset_strabs " \
  " -Dmemmove=memmove_strabs " \
  " -Dmemcmp=memcmp_strabs "

static const char *cpp_defines_16[] ={
"__INT_MAX__=32767",
"__CHAR_BIT__=8",
"__WCHAR_MAX__=32767",
"__SCHAR_MAX__=127",
"__SHRT_MAX__=32767",
"__LONG_LONG_MAX__=2147483647L",
"__LONG_MAX__=2147483647",
"__FLT_MIN__=1.17549435e-38F",
"__FLT_MAX__=3.40282347e+38F",
"__LDBL_MIN__=3.36210314311209350626e-4932L",
"__LDBL_MAX__=1.18973149535723176502e+4932L",
"__DBL_MIN__=2.2250738585072014e-308",
"__DBL_MAX__=1.7976931348623157e+308",
"__SIZE_TYPE__=\"unsigned int\"",
"__PTRDIFF_TYPE__=int",
"__WCHAR_TYPE__=int",
"__WINT_TYPE__=int",
"__INTMAX_TYPE__=\"long long int\"",
"__UINTMAX_TYPE__=\"long long unsigned int\"",
NULL
};

static const char *cpp_defines_32[] ={
"__INT_MAX__=2147483647",
"__CHAR_BIT__=8",
"__WCHAR_MAX__=2147483647",
"__SCHAR_MAX__=127",
"__SHRT_MAX__=32767",
"__LONG_LONG_MAX__=9223372036854775807LL",
"__LONG_MAX__=2147483647L",
"__FLT_MIN__=1.17549435e-38F",
"__FLT_MAX__=3.40282347e+38F",
"__LDBL_MIN__=3.36210314311209350626e-4932L",
"__LDBL_MAX__=1.18973149535723176502e+4932L",
"__DBL_MIN__=2.2250738585072014e-308",
"__DBL_MAX__=1.7976931348623157e+308",
" __SIZE_TYPE__=\"long unsigned int\"",
"__PTRDIFF_TYPE__=int",
"__WCHAR_TYPE__=int",
"__WINT_TYPE__=int",
"__INTMAX_TYPE__=\"long long int\"",
"__UINTMAX_TYPE__=\"long long unsigned int\"",
NULL
};

static const char *cpp_defines_64[] ={
"__INT_MAX__=2147483647",
"__CHAR_BIT__=8",
"__WCHAR_MAX__=2147483647",
"__SCHAR_MAX__=127",
"__SHRT_MAX__=32767",
"__LONG_LONG_MAX__=9223372036854775807LL",
"__LONG_MAX__=9223372036854775807L",
"__FLT_MIN__=1.17549435e-38F",
"__FLT_MAX__=3.40282347e+38F",
"__LDBL_MIN__=3.36210314311209350626e-4932L",
"__LDBL_MAX__=1.18973149535723176502e+4932L",
"__DBL_MIN__=2.2250738585072014e-308",
"__DBL_MAX__=1.7976931348623157e+308",
"__x86_64=1",
"__LP64__=1",
"__x86_64__=1",
"_LP64=1",
"__SIZE_TYPE__=\"long unsigned int\"",
"__PTRDIFF_TYPE__=int",
"__WCHAR_TYPE__=int",
"__WINT_TYPE__=int",
"__INTMAX_TYPE__=\"long long int\"",
"__UINTMAX_TYPE__=\"long long unsigned int\"",
NULL
};

static const char *cpp_defines_strabs[] ={
"getopt=getopt_strabs",
"fopen=fopen_strabs",
"fgets=fgets_strabs",
"fputs=fputs_strabs",
"calloc=calloc_strabs",
"atoi=atoi_strabs",
"atol=atol_strabs",
"getenv=getenv_strabs",
"strcpy=strcpy_strabs",
"strncpy=strncpy_strabs",
"strcat=strcat_strabs",
"strncat=strncat_strnabs",
"strcmp=strcmp_strabs",
"strncmp=strncmp_strabs",
"strlen=strlen_strabs",
"strdup=strdup_strabs",
"memcpy=memcpy_strabs",
"memset=memset_strabs",
"memmove=memmove_strabs",
"memcmp=memcmp_strabs",
NULL
};

void setup_cpp_defs(const char **defs)
{

  while (*defs != NULL)
    record_define(*defs);

  return;
}

bool c_preprocess(
  std::istream &instream,
  const std::string &path,
  std::ostream &outstream,
  message_handlert &message_handler)
{
  char out_file_buf[32], stderr_file_buf[32];
  pid_t pid;
  int fd, status;

  message_streamt message_stream(message_handler);

  sprintf(out_file_buf, "/tmp/ESBMC_XXXXXX");
  fd = mkstemp(out_file_buf);
  if (fd < 0) {
    message_stream.error("Couldn't open preprocessing output file");
    return true;
  }
  close(fd);

  sprintf(stderr_file_buf, "/tmp/ESBMC_XXXXXX");
  fd = mkstemp(stderr_file_buf);
  if (fd < 0) {
    message_stream.error("Couldn't open preprocessing stderr file");
    return true;
  }

  pid = fork();
  if (pid != 0) {

    close(fd);

    if (waitpid(pid, &status, 0) < 0) {
      message_stream.error("Failed to wait for preprocessing process");
      return true;
    }

    std::ifstream stderr_input(stderr_file_buf);
    message_stream.str << stderr_input;

    std::ifstream output_input(out_file_buf);
    outstream << output_input;

    unlink(stderr_file_buf);
    unlink(out_file_buf);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      message_stream.error("Preprocessing failed");
      return true;
    }

    return false;
  }

  close(STDERR_FILENO);
  dup2(fd, STDERR_FILENO);
  close(fd);

  return false;
}

#if 0
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

  // Additional cruft: Depending on what locking detection is enabled, we may
  // want to switch the behavior of pthread functions on the fly. Rather than
  // hacking that in elsewhere, we instead preprocess relevant symbol names to
  // whatever it should be.

  #ifndef _WIN32
  if (config.ansi_c.deadlock_check) {
    command+=" -Dpthread_mutex_lock=pthread_mutex_lock_check ";
  }
  if (!config.ansi_c.deadlock_check && config.ansi_c.lock_check) {
    command+=" -Dpthread_mutex_unlock=pthread_mutex_unlock_check ";
    command+=" -Dpthread_cond_wait=pthread_cond_wait_check ";
  }
  #endif

  if (config.ansi_c.string_abstraction)
    command+=STRABS_DEFS;

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
#endif
