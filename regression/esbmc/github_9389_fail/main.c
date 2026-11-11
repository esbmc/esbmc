// #9389: read() returns 0 at end-of-file.
#include <assert.h>
#include <unistd.h>

int main(void)
{
  char b[8];
  ssize_t n = read(0, b, 8);
  assert(n != 0);
  return 0;
}
