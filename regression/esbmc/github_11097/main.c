// https://github.com/esbmc/esbmc/issues/11097
#include <assert.h>
#include <errno.h>
#include <unistd.h>

int main(void)
{
  int fds[2];
  pipe(fds);
  off_t pos = lseek(fds[0], 0, -1);
  assert(pos == -1);
  assert(errno == EINVAL);
  return 0;
}
