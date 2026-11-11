// #9895: %s reads a live buffer; %p only prints a pointer, which may be dead
// or NULL. A '*' width or precision and %n each take an argument of their own.
// Past a conversion the check does not know, it stops.
#include <stdio.h>

int main()
{
  char live[4] = "abc";
  char *dead;
  {
    char buf[4] = "abc";
    dead = buf;
  }
  int k;
  printf("%s %p\n", live, (void *)dead);
  printf("%*d %p %d %s\n", 3, 1, (void *)0, 2, live);
  printf("%.*s %p %s\n", 2, live, (void *)0, live);
  printf("ab%n %p %s\n", &k, (void *)0, live);
  // %m is a glibc conversion that takes no argument.
  printf("%m %s %p\n", live, (void *)0);
  return 0;
}
