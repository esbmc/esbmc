/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <iostream>
#include <util/config.h>
#include <util/irep2.h>

configt config;

void configt::ansi_ct::set_16()
{
  bool_width=1*8;
  int_width=2*8;
  long_int_width=4*8;
  char_width=1*8;
  short_int_width=2*8;
  long_long_int_width=8*8;
  pointer_width=4*8;
  pointer_diff_width=4*8;
  single_width=4*8;
  double_width=8*8;
  long_double_width=8*8;
  char_is_unsigned=false;
  word_size=16;
  wchar_t_width=2*8;
  alignment=2;
}

void configt::ansi_ct::set_32()
{
  bool_width=1*8;
  int_width=4*8;
  long_int_width=4*8;
  char_width=1*8;
  short_int_width=2*8;
  long_long_int_width=8*8;
  pointer_width=4*8;
  pointer_diff_width=4*8;
  single_width=4*8;
  double_width=8*8;
  long_double_width=8*8;
  char_is_unsigned=false;
  word_size=32;
  wchar_t_width=4*8;
  alignment=4;
}

void configt::ansi_ct::set_64()
{
  bool_width=1*8;
  int_width=4*8;
  long_int_width=8*8;
  char_width=1*8;
  short_int_width=2*8;
  long_long_int_width=8*8;
  pointer_width=8*8;
  pointer_diff_width=8*8;
  single_width=4*8;
  double_width=8*8;
  long_double_width=16*8;
  char_is_unsigned=false;
  word_size=64;
  wchar_t_width=4*8;
  alignment=4;
}

bool configt::set(const cmdlinet &cmdline)
{
  // defaults
  ansi_c.set_32();

  #ifdef HAVE_FLOATBV
  ansi_c.use_fixed_for_float=false;
  #else
  ansi_c.use_fixed_for_float=true;
  #endif

  ansi_c.endianess=ansi_ct::NO_ENDIANESS;
  ansi_c.os=ansi_ct::NO_OS;
  ansi_c.lib=configt::ansi_ct::LIB_NONE;
  ansi_c.rounding_mode=ieee_floatt::ROUND_TO_EVEN;

  if(cmdline.isset("16"))
    ansi_c.set_16();

  if(cmdline.isset("32"))
    ansi_c.set_32();

  if(cmdline.isset("64"))
    ansi_c.set_64();

  if(cmdline.isset("function"))
    main=cmdline.getval("function");

  if(cmdline.isset('D'))
    ansi_c.defines=cmdline.get_values('D');

  if(cmdline.isset('I'))
    ansi_c.include_paths=cmdline.get_values('I');

  if(cmdline.isset("floatbv") && cmdline.isset("fixedbv"))
  {
    std::cerr << "Can't set both floatbv and fixedbv modes" << std::endl;
    return true;
  }

  if(cmdline.isset("floatbv"))
    ansi_c.use_fixed_for_float=false;

  if(cmdline.isset("fixedbv"))
    ansi_c.use_fixed_for_float=true;

  if(cmdline.isset("i386-linux"))
  {
    ansi_c.os=configt::ansi_ct::OS_I386_LINUX;
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
  }

  if(cmdline.isset("i386-win32"))
  {
    ansi_c.os=configt::ansi_ct::OS_WIN32;
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
  }

  if(cmdline.isset("i386-macos"))
  {
    ansi_c.os=configt::ansi_ct::OS_I386_MACOS;
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
  }

  if(cmdline.isset("ppc-macos"))
  {
    ansi_c.os=configt::ansi_ct::OS_PPC_MACOS;
    ansi_c.endianess=configt::ansi_ct::IS_BIG_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
  }

  if(cmdline.isset("no-arch"))
  {
    ansi_c.os=configt::ansi_ct::NO_OS;
    ansi_c.endianess=configt::ansi_ct::NO_ENDIANESS;
    ansi_c.lib=configt::ansi_ct::LIB_NONE;
  }
  else if(ansi_c.os==configt::ansi_ct::NO_OS)
  {
    // this is the default
    #ifdef _WIN32
    ansi_c.os=configt::ansi_ct::OS_WIN32;
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
    #else
    #ifdef __APPLE__
    ansi_c.os=configt::ansi_ct::OS_I386_MACOS;
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
    #else
    ansi_c.os=configt::ansi_ct::OS_I386_LINUX;
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;
    ansi_c.lib=configt::ansi_ct::LIB_FULL;
    #endif
    #endif
  }

  if(cmdline.isset("no-library"))
    ansi_c.lib=configt::ansi_ct::LIB_NONE;

  if(cmdline.isset("little-endian"))
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;

  if(cmdline.isset("big-endian"))
    ansi_c.endianess=configt::ansi_ct::IS_BIG_ENDIAN;

  if(cmdline.isset("round-to-even") || cmdline.isset("round-to-nearest"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_EVEN;

  if(cmdline.isset("round-to-plus-inf"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_PLUS_INF;

  if(cmdline.isset("round-to-minus-inf"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_MINUS_INF;

  if(cmdline.isset("round-to-zero"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_ZERO;

  if(cmdline.isset("little-endian") && cmdline.isset("big-endian"))
  {
    std::cerr << "Can't set both little and big endian modes" << std::endl;
    return true;
  }

  if(cmdline.isset("unsigned-char"))
    ansi_c.char_is_unsigned=true;

  return false;
}

std::string configt::this_architecture()
{
  std::string this_arch;

  // following http://wiki.debian.org/ArchitectureSpecificsMemo

  #ifdef __alpha__
  this_arch="alpha";
  #elif __armel__
  this_arch="armel";
  #elif __aarch64__
  this_arch="arm64";
  #elif __arm__
    #ifdef __ARM_PCS_VFP
    this_arch="armhf"; // variant of arm with hard float
    #else
    this_arch="arm";
    #endif
  #elif __mipsel__
    #if _MIPS_SIM==_ABIO32
    this_arch="mipsel";
    #elif _MIPS_SIM==_ABIN32
    this_arch="mipsn32el";
    #else
    this_arch="mips64el";
    #endif
  #elif __mips__
    #if _MIPS_SIM==_ABIO32
    this_arch="mips";
    #elif _MIPS_SIM==_ABIN32
    this_arch="mipsn32";
    #else
    this_arch="mips64";
    #endif
  #elif __powerpc__
    #if defined(__ppc64__) || defined(__PPC64__) || defined(__powerpc64__) || defined(__POWERPC64__)
      #ifdef __LITTLE_ENDIAN__
      this_arch="ppc64le";
      #else
      this_arch="ppc64";
      #endif
    #else
    this_arch="powerpc";
    #endif
  #elif __sparc__
    #ifdef __arch64__
    this_arch="sparc64";
    #else
    this_arch="sparc";
    #endif
  #elif __ia64__
  this_arch="ia64";
  #elif __s390x__
  this_arch="s390x";
  #elif __s390__
  this_arch="s390";
  #elif __x86_64__
    #ifdef __ILP32__
    this_arch="x32"; // variant of x86_64 with 32-bit pointers
    #else
    this_arch="x86_64";
    #endif
  #elif __i386__
  this_arch="i386";
  #elif _WIN64
  this_arch="x86_64";
  #elif _WIN32
  this_arch="i386";
  #else
  // something new and unknown!
  this_arch="unknown";
  #endif

  return this_arch;
}

/*******************************************************************\

Function: configt::ansi_ct::this_operating_system

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string configt::this_operating_system()
{
  std::string this_os;

  #ifdef _WIN32
  this_os="windows";
  #elif __APPLE__
  this_os="macos";
  #elif __FreeBSD__
  this_os="freebsd";
  #elif __linux__
  this_os="linux";
  #elif __SVR4
  this_os="solaris";
  #else
  this_os="unknown";
  #endif

  return this_os;
}
