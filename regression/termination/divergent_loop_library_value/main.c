/* getpid() never returns 0, so the loop exits at once, but termination is
 * not proved: ESBMC treats getpid() as returning any value. */
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
