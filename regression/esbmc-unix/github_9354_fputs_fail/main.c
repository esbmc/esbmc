/* #9354: fputs reads its argument up to the terminating null character. */
#include <stdio.h>

int main(void)
{
  char tag[3] = {'a', 'b', 'c'};
  fputs(tag, stdout);
  return 0;
}
