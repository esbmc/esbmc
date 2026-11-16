// #9389: read() returns a count between 0 (end-of-file) and nbyte.
#include <assert.h>
#include <unistd.h>

int main(void)
{
  char b[8];
  ssize_t n = read(0, b, 8);
  assert(n >= 0 && n <= 8);
  if (n > 0)
    b[n - 1] = 0;
  return 0;
}
