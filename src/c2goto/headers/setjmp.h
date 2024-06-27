
#pragma once

#include_next <setjmp.h>

/* mingw64 at least since v7 defines _setjmp to the incompatible
 * __intrinsic_setjmpex. We don't care. */
#undef _setjmp
