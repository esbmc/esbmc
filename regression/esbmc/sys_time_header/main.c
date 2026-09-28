#include <sys/select.h>
#include <sys/time.h>
#include <assert.h>

int main()
{
  /* Including both headers used to fail to compile on macOS: the platform
     <sys/time.h> redefined the fd_set and timeval declared here. */
  fd_set s;
  FD_ZERO(&s);
  assert(!FD_ISSET(3, &s));
  FD_SET(3, &s);
  assert(FD_ISSET(3, &s));
  FD_CLR(3, &s);
  assert(!FD_ISSET(3, &s));

  struct timeval a, b;
  a.tv_sec = 1;
  a.tv_usec = 0;
  b.tv_sec = 2;
  b.tv_usec = 0;
  assert(timercmp(&a, &b, <));

  timerclear(&a);
  assert(!timerisset(&a));
  a.tv_usec = 5;
  assert(timerisset(&a));
  return 0;
}
