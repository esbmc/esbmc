#ifndef CPROVER_UTIL_CONFIG_H
#define CPROVER_UTIL_CONFIG_H

#include <unordered_set>
#include <util/cmdline.h>
#include <util/options.h>
#include <langapi/mode.h>

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
  struct triple
  {
    std::string arch = "none";
    std::string vendor = "unknown";
    std::string os = "elf";
    std::string flavor;

    bool is_windows_abi() const;
    bool is_freebsd() const;
    bool is_macos() const;
    std::string to_string() const;
  };

#define dm(char, short, int, long, addr, word, long_dbl)                       \
  ((uint64_t)(char) | (uint64_t)(short) << 8 | (uint64_t)(int) << 16 |         \
   (uint64_t)(long) << 24 | (uint64_t)(addr) << 32 | (uint64_t)(word) << 40 |  \
   (uint64_t)(long_dbl) << 48)

  enum data_model : uint64_t
  {
    /* 16-bit */
    IP16 = dm(8, 16, 16, 32, 16, 16, 64), /* unsegmented 16-bit */
    LP32 = dm(8, 16, 16, 32, 32, 16, 64), /* segmented 16-bit DOS, Win16 */
    /* 32-bit */
    IP32 = dm(8, 16, 32, 64, 32, 32, 96),  /* Ultrix '82-'95 */
    ILP32 = dm(8, 16, 32, 32, 32, 32, 96), /* Win32 || other 32-bit Unix */
    /* 64-bit */
    LLP64 = dm(8, 16, 32, 32, 64, 64, 128), /* Win64 */
    ILP64 = dm(8, 16, 64, 64, 64, 64, 128), /* Unicos for Cray PVP systems */
    LP64 = dm(8, 16, 32, 64, 64, 64, 128),  /* other 64-bit Unix */
  };

#undef dm

  // Language the frontend has been parsing
  language_idt language = language_idt::NONE;

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

    // for fixed size
    unsigned int_128_width = 128;

    typedef enum
    {
      NO_ENDIANESS,
      IS_LITTLE_ENDIAN,
      IS_BIG_ENDIAN
    } endianesst;
    endianesst endianess;

    triple target;

    std::list<std::string> defines;
    std::list<std::string> include_paths;
    std::list<std::string> idirafter_paths;
    std::list<std::string> forces;
    std::list<std::string> warnings;

    typedef enum
    {
      LIB_NONE,
      LIB_FULL
    } libt;
    libt lib;

    void set_data_model(enum data_model dm);
  } ansi_c;

  std::string main;
  std::unordered_set<std::string> no_slice_names;
  std::unordered_set<std::string> no_slice_ids;

  bool set(const cmdlinet &cmdline);

  optionst options;

  static std::string this_architecture();
  static std::string this_operating_system();

  static triple host();
};

extern configt config;

#endif
