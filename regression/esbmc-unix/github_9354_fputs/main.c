/* #9354: fputs and fputs_unlocked of terminated strings. */
#include <stdio.h>

int main(void)
{
  char tag[4] = {'a', 'b', 'c', '\0'};
  fputs(tag, stdout);
  fputs_unlocked(tag + 3, stdout);
  return 0;
}
