#include <stdio.h>

/* Provide concrete definitions for stdin, stdout, stderr.
 *
 * The C standard requires these to be macros (C99 §7.19.1), but the actual
 * objects they expand to must be defined somewhere.  On Linux/glibc that is
 * libio/stdio.c; on macOS that is in libSystem.  ESBMC bundles its own libc,
 * so we provide the definitions here.
 *
 * The macro trick:
 *   - On macOS,  stdin expands to __stdinp  (from <stdio.h>)
 *   - On Linux,  stdin expands to stdin     (identity macro in glibc)
 * Either way, writing `FILE *stdin = ...` after including <stdio.h> defines
 * exactly the symbol that extern declarations in user code refer to.
 */

static FILE __esbmc_stdin_obj;
static FILE __esbmc_stdout_obj;
static FILE __esbmc_stderr_obj;

FILE *stdin = &__esbmc_stdin_obj;
FILE *stdout = &__esbmc_stdout_obj;
FILE *stderr = &__esbmc_stderr_obj;

/* sys_nerr and sys_errlist: legacy BSD error-reporting globals declared as
 * extern in <stdio.h> on BSD/macOS.  ESBMC's strerror() uses its own local
 * copy, so the values here are not significant for verification. */
const int sys_nerr = 0;
const char *const sys_errlist[1] = {(const char *)0};
