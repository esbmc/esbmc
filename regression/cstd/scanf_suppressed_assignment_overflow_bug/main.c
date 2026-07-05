#include <stdio.h>

// A "%*d" assignment-suppressed directive (and a "%%" literal) consumes no
// argument. The scanf overflow check must not let it misalign the format
// directives against the argument list and abandon the check -- the genuine
// unbounded "%s" read into buf[4] must still be reported as a buffer overflow.
int main(void)
{
  char buf[4];
  scanf("%*d %s", buf);
  return 0;
}
