#include <stdio.h>

// A width-less wide-string conversion (%ls, and the XSI/glibc %S alias) reads an
// unbounded run of bytes into the caller's buffer, exactly like %s, so it must
// still be reported as a buffer overflow. Locks in the soundness edge of the
// #1470 follow-up fix: only %lc / %C (single wide char) are bounded, not %ls.
int main(void)
{
  char buf[4];
  scanf("%ls", buf);
  return 0;
}
