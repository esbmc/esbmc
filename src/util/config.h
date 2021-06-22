/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_UTIL_CONFIG_H
#define CPROVER_UTIL_CONFIG_H

#include <util/cmdline.h>
#include <util/options.h>

#ifndef GNUC_PREREQ
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && defined(__GNUC_PATCHLEVEL__)
#define GNUC_PREREQ(maj, min, patch)                                           \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) + __GNUC_PATCHLEVEL__ >=          \
   ((maj) << 20) + ((min) << 10) + (patch))
#elif defined(__GNUC__) && defined(__GNUC_MINOR__)
#define GNUC_PREREQ(maj, min, patch)                                           \
  ((__GNUC__ << 20) + (__GNUC_MINOR__ << 10) >= ((maj) << 20) + ((min) << 10))
#else
#define GNUC_PREREQ(maj, min, patch) 0
#endif
#endif

#if __has_attribute(noinline) || GNUC_PREREQ(3, 4, 0)
#define ATTRIBUTE_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define ATTRIBUTE_NOINLINE __declspec(noinline)
#else
#define ATTRIBUTE_NOINLINE
#endif

#if __has_attribute(used) || GNUC_PREREQ(3, 1, 0)
#define ATTRIBUTE_USED __attribute__((__used__))
#else
#define ATTRIBUTE_USED
#endif

#if !defined(NDEBUG)
#define DUMP_METHOD ATTRIBUTE_NOINLINE ATTRIBUTE_USED
#else
#define DUMP_METHOD ATTRIBUTE_NOINLINE
#endif

class configt
{
public:
  struct ansi_ct
  {
    // for ANSI-C
    unsigned int_width;
    unsigned long_int_width;
    unsigned bool_width;
    unsigned char_width;
    unsigned short_int_width;
    unsigned long_long_int_width;
    unsigned pointer_width;
    unsigned single_width;
    unsigned double_width;
    unsigned long_double_width;
    unsigned pointer_diff_width;
    unsigned word_size;
    unsigned wchar_t_width;

    bool char_is_unsigned;
    bool use_fixed_for_float;

    void set_16();
    void set_32();
    void set_64();

    // alignment (in structs) measured in bytes
    unsigned alignment;

    typedef enum
    {
      NO_ENDIANESS,
      IS_LITTLE_ENDIAN,
      IS_BIG_ENDIAN
    } endianesst;
    endianesst endianess;

    typedef enum
    {
      NO_OS,
      OS_I386_LINUX,
      OS_I386_MACOS,
      OS_PPC_MACOS,
      OS_WIN32
    } ost;
    ost os;

    std::list<std::string> defines;
    std::list<std::string> include_paths;
    std::list<std::string> forces;
    std::list<std::string> warnings;

    typedef enum
    {
      LIB_NONE,
      LIB_FULL
    } libt;
    libt lib;
  } ansi_c;

  std::string main;

  bool set(const cmdlinet &cmdline, const messaget &msg);

  optionst options;

  static std::string this_architecture();
  static std::string this_operating_system();
};

extern configt config;

#endif
