// #10303: printf and fprintf read a live heap format string, then free it, and
// a format chosen between two literals.
#include <stdio.h>
#include <stdlib.h>

int nondet_int(void);

int main()
{
  char *p = (char *)malloc(4);
  if (!p)
    return 0;
  p[0] = '%';
  p[1] = 'd';
  p[2] = '\n';
  p[3] = '\0';
  printf(p, 1);
  fprintf(stdout, p, 2);
  free(p);
  const char *f = nondet_int() ? "%d\n" : "x\n";
  printf(f, 3);
  return 0;
}
