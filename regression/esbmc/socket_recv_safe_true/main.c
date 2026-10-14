/*
 * socket_recv_safe_true — VERIFICATION SUCCESSFUL expected
 *
 * Same recv() pattern as socket_recv_overflow_false, but the caller
 * correctly guards the index: buf[n-1] is only accessed when n >= 1
 * and n <= BUF_SIZE, which is always satisfied because the recv()
 * model constrains its return value to [1, len].
 *
 * Expected: ESBMC reports no violations.
 */

#include <stdint.h>
#include <sys/socket.h>

#define BUF_SIZE 16

int main(void)
{
  int fd = socket(AF_INET, SOCK_STREAM, 0);

  uint8_t buf[BUF_SIZE];
  ssize_t n = recv(fd, buf, BUF_SIZE, 0);

  if (n > 0 && (size_t)n <= BUF_SIZE)
  {
    /* Safe: index is within [0, BUF_SIZE - 1] */
    uint8_t last = buf[n - 1];
    (void)last;
  }

  return 0;
}
