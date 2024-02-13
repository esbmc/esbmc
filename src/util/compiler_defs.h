#pragma once

/// Header to define some properties that are OS or Compiler specific
/* Warnings / Diagnostics related
  These warnings come from Clang, so it might
  have a slight difference between Clang 11 and 14 warnings
*/
#if defined(_MSC_VER)
#define CC_DIAGNOSTIC_PUSH() __pragma(warning(push))
#define CC_DIAGNOSTIC_POP() __pragma(warning(pop))
#define CC_DIAGNOSTIC_DISABLE(x) __pragma(warning(disable : x))

// NOTE: The magic numbers came directly from MSVC
#define CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()                                     \
  __pragma(message("Disabling diagnostic checks for clang headers"))           \
    CC_DIAGNOSTIC_DISABLE(4146 4244 4267 4291 4624)
#else
#define CC_DIAGNOSTIC_PUSH() _Pragma("GCC diagnostic push")
#define CC_DIAGNOSTIC_POP() _Pragma("GCC diagnostic pop")
#define DO_PRAGMA(x) _Pragma(#x)

#define CC_DIAGNOSTIC_IGNORE_LLVM_CHECKS()                                     \
  DO_PRAGMA(GCC diagnostic ignored "-Wstrict-aliasing")                        \
  DO_PRAGMA(GCC diagnostic ignored "-Wunused-parameter")                       \
  DO_PRAGMA(GCC diagnostic ignored "-Wnonnull")                                \
  DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#endif

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

#if defined(_MSC_VER)
#define ATTRIBUTE_NOINLINE __declspec(noinline)
#elif __has_attribute(noinline) || GNUC_PREREQ(3, 4, 0)
#define ATTRIBUTE_NOINLINE __attribute__((noinline))
#else
#define ATTRIBUTE_NOINLINE
#endif

#define ATTRIBUTE_USED
#ifndef _MSC_VER
#if GNUC_PREREQ(3, 1, 0) || __has_attribute(used)
#undef ATTRIBUTE_USED
#define ATTRIBUTE_USED __attribute__((__used__))
#endif
#endif

#if !defined(NDEBUG)
#define DUMP_METHOD ATTRIBUTE_NOINLINE ATTRIBUTE_USED
#else
#define DUMP_METHOD ATTRIBUTE_NOINLINE
#endif
