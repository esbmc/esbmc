/*
 * socket_ssize_t_ilp32_false — VERIFICATION FAILED expected
 *
 * Same ssize_t pin as socket_ssize_t_ilp32_true, but the caller assumes
 * send() always succeeds. The model may return -1, so the assertion fails.
 */

#include <assert.h>
#include <sys/socket.h>

#ifdef __APPLE__
_Static_assert(
  _Generic((ssize_t)0, long : 1, default : 0),
  "ssize_t must be long on Darwin");
#else
_Static_assert(
  _Generic((ssize_t)0, int : 1, default : 0),
  "ssize_t must be int on ILP32");
#endif

#define BUF_SIZE 4

int main(void)
{
  int fd = socket(AF_INET, SOCK_STREAM, 0);

  char buf[BUF_SIZE] = {0};
  ssize_t n = send(fd, buf, BUF_SIZE, 0);

  assert(n >= 0);

  return 0;
}
