/* github #10303: printf and fprintf did not check that the format pointer
 * is valid. */
#include <stdio.h>
#include <stdlib.h>

int nondet_int(void);

int main()
{
  char *p = malloc(3);
  if (!p)
    return 0;
  p[0] = 'h';
  p[1] = '\n';
  p[2] = '\0';
  int choice = nondet_int();
  if (choice)
    free(p);
  printf(p);
  fprintf(stdout, p);
  if (!choice)
    free(p);
  return 0;
}
