#pragma once

// libc++ ships its own <uchar.h>, which -nostdinc++ removes. Darwin has no C
// <uchar.h>, and under --mix-cpp-host-headers the next one found is libc++'s,
// so only elsewhere does this reach the C library's declarations.
#if __has_include_next(<uchar.h>) && !defined(__APPLE__)
#  define __ESBMC_C_UCHAR_H
#  include_next <uchar.h>
#else
#  include <wchar.h>
#endif
