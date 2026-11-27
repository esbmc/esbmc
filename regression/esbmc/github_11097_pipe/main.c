#include <assert.h>
#include <errno.h>
#include <unistd.h>

int main(void)
{
  int fds[2];
  pipe(fds);
  off_t pos = lseek(fds[0], 0, SEEK_SET);
  assert(pos == -1);
  assert(errno == ESPIPE);
  return 0;
}
