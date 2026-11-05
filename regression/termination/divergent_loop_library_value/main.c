/* getpid() has no body in ESBMC, but its result is not the program's to
 * choose: a process id is never 0, so the loop exits at once. */
#include <unistd.h>

int main(void)
{
  while (1)
  {
    int r = getpid();
    if (r == 0)
      continue;
    break;
  }
  return 0;
}
