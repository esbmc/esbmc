#include <stdio.h>

// A width-less string conversion (%s) reads an unbounded run of bytes into the
// caller's buffer; this must still be reported as a buffer overflow. Guards
// against the #1470 follow-up fix over-suppressing genuine string overflows.
int main(void)
{
  char buf[4];
  scanf("%s", buf);
  return 0;
}
