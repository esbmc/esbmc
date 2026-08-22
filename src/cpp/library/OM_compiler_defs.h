// Compiler diagnostic tweaks and C++03/C++11 compatibility macros for OMs
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
#  define OM_NULLPTR nullptr
#else
#  define OM_CONSTEXPR
#  define OM_NOEXCEPT
#  define OM_NULLPTR 0
#endif

// char_traits members became constexpr in C++17 (P0426R1); libc++ gates them
// the same way, as _LIBCPP_CONSTEXPR_SINCE_CXX17. Starting earlier would let a
// constexpr string_view built from a literal compile here but not under clang.
#if __cplusplus >= 201703L
#  define OM_CONSTEXPR17 constexpr
#else
#  define OM_CONSTEXPR17
#endif

// A static data member with an in-class initialiser. OM_CONSTEXPR alone is not
// enough: eliding `constexpr` would leave a non-const member, which C++03 does
// not let you initialise in-class. Only valid for integral and enum types.
#if __cplusplus >= 201103L
#  define OM_STATIC_CONSTANT static constexpr
#else
#  define OM_STATIC_CONSTANT static const
#endif
