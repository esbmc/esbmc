#pragma once

/// Header to define some properties that are OS or Compiler specific

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
