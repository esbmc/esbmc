#include <assert.h>
#include <errno.h>
#include <unistd.h>

int main(void)
{
  int fds[2];
  pipe(fds);
  lseek(fds[0], 0, -1);
  assert(errno == ESPIPE);
  return 0;
}
