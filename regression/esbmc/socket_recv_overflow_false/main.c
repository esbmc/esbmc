/*
 * socket_recv_overflow_false — VERIFICATION FAILED expected
 *
 * Models a typical network parser pattern: read bytes via recv() into a
 * fixed-size buffer, then use the returned length as an array index without
 * checking it against the buffer size. The recv() operational model returns
 * a non-deterministic length in [1, len]. When len == BUF_SIZE, a caller
 * that indexes buf[n] where n == len triggers an out-of-bounds access
 * (off-by-one).
 *
 * Expected: ESBMC reports array bounds violation.
 */

#include <stdint.h>
#include <sys/socket.h>

#define BUF_SIZE 16

int main(void)
{
  int fd = socket(AF_INET, SOCK_STREAM, 0);

  uint8_t buf[BUF_SIZE];
  ssize_t n = recv(fd, buf, BUF_SIZE, 0);

  if (n > 0)
  {
    /* BUG: uses n as index without checking n < BUF_SIZE */
    uint8_t last = buf[n]; /* out-of-bounds when n == BUF_SIZE */
    (void)last;
  }

  return 0;
}
