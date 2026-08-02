/* #5565: fclose() on a standard stream must not report "invalid pointer
 * freed".  stdin/stdout/stderr are not heap objects, but the fclose model
 * called free(stream) unconditionally (outside SV-COMP), so closing stdout --
 * exactly what coreutils' close_stdout atexit handler does -- raised a
 * spurious "invalid pointer freed".  fclose(stdout) is well-defined C. */
#include <stdio.h>

int main(void)
{
  fclose(stdout);
  fclose(stderr);
  return 0;
}
