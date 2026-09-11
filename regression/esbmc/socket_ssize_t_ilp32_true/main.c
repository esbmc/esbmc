/*
 * socket_ssize_t_ilp32_true — VERIFICATION SUCCESSFUL expected
 *
 * Pins the type the socket model gives ssize_t. It has to be the one the host
 * C library uses, or send/recv end up with two incompatible declarations once
 * c2goto merges the model translation units (-Wodr). That clash is a build-
 * time property of c2goto and cannot be observed from a test.desc, so this
 * pins the typedef it comes from: pointer-width under glibc's ILP32, where
 * <sys/socket.h> used to say `long`.
 *
 * send() returns -1 or a count in [0, len], so the bound below always holds.
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

  assert(n <= (ssize_t)BUF_SIZE);

  return 0;
}
