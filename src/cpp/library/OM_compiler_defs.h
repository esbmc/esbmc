// Header to define some compmiler diagnostic tweaks for OMs
#pragma once

#define DO_PRAGMA(x) _Pragma(#x)
#define CC_DIAGNOSTIC_PUSH() _Pragma("GCC diagnostic push")
#define CC_DIAGNOSTIC_POP() _Pragma("GCC diagnostic pop")

#define CC_DIAGNOSTIC_IGNORE_OM_LLVM_CHECKS()                                  \
  DO_PRAGMA(GCC diagnostic ignored "-Wreturn-type")
