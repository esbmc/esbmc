/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifdef _WIN32
#include <windows.h>
#undef small
#undef ERROR
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>

#include <config.h>
#include <i2string.h>
#include <message_stream.h>

#include "c_preprocess.h"

extern "C" {
#include "cpp/iface.h"
}

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
"__SIZE_TYPE__=unsigned int",
"__PTRDIFF_TYPE__=int",
"__WCHAR_TYPE__=int",
"__WINT_TYPE__=int",
"__INTMAX_TYPE__=long long int",
"__UINTMAX_TYPE__=long long unsigned int",
"__WORDSIZE=16",
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
"__SIZE_TYPE__=long unsigned int",
"__PTRDIFF_TYPE__=int",
"__WCHAR_TYPE__=int",
"__WINT_TYPE__=int",
"__INTMAX_TYPE__=long long int",
"__UINTMAX_TYPE__=long long unsigned int",
"__WORDSIZE=32",
"_X86_",
"__i386__",
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
"__SIZE_TYPE__=long unsigned int",
"__PTRDIFF_TYPE__=int",
"__WCHAR_TYPE__=int",
"__WINT_TYPE__=int",
"__INTMAX_TYPE__=long long int",
"__UINTMAX_TYPE__=long long unsigned int",
"__WORDSIZE=64",
"__x86_64",
"__x86_64__",
NULL
};

static const char *cpp_normal_defs[] = {
"__ESBMC__",
"__STRICT_ANSI__=1",
"_POSIX_SOURCE=1",
// mingw sched/pthread headers choke upon this.
// Mac's attempt to spam inline assembly everywhere if this is defined, to
// alias some deprecated function symbols
#if !defined(__WIN32__) && !defined(__APPLE__)
"_POSIX_C_SOURCE=200112L",
#endif
"__GNUC__",
"__VERIFIER_ASSUME=__ESBMC_assume",
"__VERIFIER_assume=__ESBMC_assume",
"__VERIFIER_atomic_begin=__ESBMC_atomic_begin",
"__VERIFIER_atomic_end=__ESBMC_atomic_end",
NULL
};

static const char *cpp_linux_defs[] = {
"i386",
"__i386",
"__i386__",
"linux",
"__linux",
"__linux__",
"__gnu_linux__",
"unix",
"__unix",
"__unix__",
"__null=0",
"__restrict__=/**/",
"__restrict=/**/",
NULL
};

#ifdef __APPLE__
static const char *cpp_mac_defs[] = {
"__APPLE__",
"__GNUC__",
"__asm(x)= ",
"_FORTIFY_SOURCE=0",
NULL,
};
#endif

#ifdef WIN32
static const char *cpp_windows_defs[] = {
"_WIN32",
"__restrict__=/**/",
"__restrict=/**/",
NULL
};
#endif

static const char *cpp_ansic_defs[] = {
"__STDC_VERSION__=199901L",
"__STDC_IEC_559__=1",
"__STDC_IEC_559_COMPLEX__=1",
"__STDC__",
NULL
};

static const char *cpp_cpp_defs[] = {
"__cplusplus=1",
NULL
};

int configure_and_run_cpp(const char *out_file_buf, std::string path,
		          const char **platformdefs, bool is_cpp);

void setup_cpp_defs(const char **defs)
{

  while (*defs != NULL) {
    record_define(*defs);
    defs++;
  }

  return;
}

#ifndef _WIN32

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

bool c_preprocess(
  const std::string &path,
  std::ostream &outstream,
  bool is_cpp,
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

    // Loop around this, because while under gdb, macs for some reason receive
    // a lot of EINTRs without restarting the syscall. Hmm.
    pid_t foo;
    while ((foo = waitpid(pid, &status, 0)) == -1 && errno == EINTR)
      ;
    if ((foo) < 0) {
      message_stream.error("Failed to wait for preprocessing process");
      return true;
    }

    std::ifstream stderr_input(stderr_file_buf);
    message_stream.str << stderr_input.rdbuf();
    message_stream.status();

    std::ifstream output_input(out_file_buf);
    outstream << output_input.rdbuf();

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


  const char **defs;
#ifdef __APPLE__
  defs = cpp_mac_defs;
#else
  defs = cpp_linux_defs;
#endif

  exit(configure_and_run_cpp(out_file_buf, path, defs, is_cpp));
}

#else /* __WIN32__ */

#include <io.h>

bool c_preprocess(
  const std::string &path,
  std::ostream &outstream,
  bool is_cpp,
  message_handlert &message_handler)
{
  int err, ret;
  char out_file_buf[288], tmpdir[256];

  // For Windows, we can't fork and run the preprocessor in a seperate process.
  // Instead, just run it within the existing ESBMC process.

  message_streamt message_stream(message_handler);

  GetTempPath(sizeof(tmpdir), tmpdir);
  GetTempFileName(tmpdir, "bmc", 0, out_file_buf);

  ret = configure_and_run_cpp(out_file_buf, path, cpp_windows_defs, is_cpp);
  if (ret != 0) {
    message_stream.error("Preprocessor returned an error");
    return true;
  }

  std::ifstream output_input(out_file_buf);
  outstream << output_input.rdbuf();
  output_input.close();
  DeleteFile(out_file_buf);

  cpp_clear(); // Reset cpp state

  return false;
}

#endif

int
configure_and_run_cpp(const char *out_file_buf, std::string path,
		      const char **platform_defs, bool is_cpp)
{
  int ret;

  if(config.ansi_c.word_size==16)
    setup_cpp_defs(cpp_defines_16);
  else if(config.ansi_c.word_size==32)
    setup_cpp_defs(cpp_defines_32);
  else if(config.ansi_c.word_size==64)
    setup_cpp_defs(cpp_defines_64);
  else
    std::cerr << "Bad word size " << config.ansi_c.word_size << std::endl;

  setup_cpp_defs(cpp_normal_defs);
  setup_cpp_defs(platform_defs);
  setup_cpp_defs(cpp_ansic_defs);

  if (is_cpp)
    setup_cpp_defs(cpp_cpp_defs);

  if (config.options.get_bool_option("deadlock-check")) {
    record_define("pthread_mutex_lock=pthread_mutex_lock_check");
    record_define("pthread_mutex_unlock=pthread_mutex_unlock_check");
    record_define("pthread_cond_wait=pthread_cond_wait_check");
    record_define("pthread_join=pthread_join_switch");
  } else if (config.options.get_bool_option("lock-order-check")){
    record_define("pthread_join=pthread_join_noswitch");
    record_define("pthread_mutex_lock=pthread_mutex_lock_nocheck");
    record_define("pthread_mutex_unlock=pthread_mutex_unlock_nocheck");
    record_define("pthread_cond_wait=pthread_cond_wait_nocheck");
  } else {
    record_define("pthread_join=pthread_join_noswitch");
    record_define("pthread_mutex_lock=pthread_mutex_lock_noassert");
    record_define("pthread_mutex_unlock=pthread_mutex_unlock_noassert");
    record_define("pthread_cond_wait=pthread_cond_wait_nocheck");
  }

  for(std::list<std::string>::const_iterator
      it=config.ansi_c.defines.begin();
      it!=config.ansi_c.defines.end();
      it++)
    record_define(it->c_str());

  for(std::list<std::string>::const_iterator
      it=config.ansi_c.include_paths.begin();
      it!=config.ansi_c.include_paths.end();
      it++)
    record_include(it->c_str());

  record_include("/usr/include");
  record_builtin_macros();

  if (open_output_file(out_file_buf) < 0) {
    perror("cpp couldn't open output file");
    exit(1);
  }

  ret = pushfile2(path.c_str(), path.c_str(), 0, NULL);
  fin();

  return ret;
}
