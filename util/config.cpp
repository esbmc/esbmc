/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "config.h"

configt config;

/*******************************************************************\

Function: configt::ansi_ct::set_16

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void configt::ansi_ct::set_16()
{
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

/*******************************************************************\

Function: configt::ansi_ct::set_32

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void configt::ansi_ct::set_32()
{
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
  wchar_t_width=2*8;
  alignment=4;
}

/*******************************************************************\

Function: configt::ansi_ct::set_64

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void configt::ansi_ct::set_64()
{
  int_width=4*8;
  long_int_width=8*8;
  char_width=1*8;
  short_int_width=2*8;
  long_long_int_width=16*8;
  pointer_width=8*8;
  pointer_diff_width=8*8;
  single_width=4*8;
  double_width=8*8;
  long_double_width=8*8;
  char_is_unsigned=false;
  word_size=64;
  wchar_t_width=2*8;
  alignment=4;
}

/*******************************************************************\

Function: configt::ansi_ct::set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

  if(cmdline.isset("string-abstraction"))
    ansi_c.string_abstraction=true;
  else
    ansi_c.string_abstraction=false;

  if(cmdline.isset("no-lock-check"))
    ansi_c.lock_check=false;
  else
	ansi_c.lock_check=true;

  if(cmdline.isset("deadlock-check"))
    ansi_c.deadlock_check=true;
  else
    ansi_c.deadlock_check=false;

#if 0
  if (cmdline.isset("uw-model"))
  {
    ansi_c.deadlock_check=false;
    ansi_c.lock_check=false;
  }
#endif

  if(cmdline.isset("no-library"))
    ansi_c.lib=configt::ansi_ct::LIB_NONE;

  if(cmdline.isset("little-endian"))
    ansi_c.endianess=configt::ansi_ct::IS_LITTLE_ENDIAN;

  if(cmdline.isset("big-endian"))
    ansi_c.endianess=configt::ansi_ct::IS_BIG_ENDIAN;

  if(cmdline.isset("little-endian") &&
     cmdline.isset("big-endian")) {
      std::cerr << "Can't set both little and big endian modes" << std::endl;
      return true;
  }

  if(cmdline.isset("unsigned-char"))
    ansi_c.char_is_unsigned=true;

  if(cmdline.isset("round-to-even") ||
     cmdline.isset("round-to-nearest"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_EVEN;

  if(cmdline.isset("round-to-plus-inf"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_PLUS_INF;

  if(cmdline.isset("round-to-minus-inf"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_MINUS_INF;

  if(cmdline.isset("round-to-zero"))
    ansi_c.rounding_mode=ieee_floatt::ROUND_TO_ZERO;

  return false;
}

