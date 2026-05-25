// Header to define some compmiler diagnostic tweaks for OMs
#pragma once

#define DO_PRAGMA(x) _Pragma(#x)
#define CC_DIAGNOSTIC_PUSH() _Pragma("GCC diagnostic push")
#define CC_DIAGNOSTIC_POP() _Pragma("GCC diagnostic pop")

#define CC_DIAGNOSTIC_IGNORE_OM_LLVM_CHECKS()                                  \
  DO_PRAGMA(GCC diagnostic ignored "-Wreturn-type")

// C++11+ syntax that is invalid under --std c++03 — elide it on the C++03 path.
#if __cplusplus >= 201103L
#  define OM_CONSTEXPR constexpr
#  define OM_NOEXCEPT noexcept
#else
#  define OM_CONSTEXPR
#  define OM_NOEXCEPT
#endif
