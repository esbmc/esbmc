#include <sys/select.h>
#include <sys/time.h>
#include <assert.h>

int main()
{
  fd_set s;
  FD_ZERO(&s);
  FD_SET(3, &s);
  assert(FD_ISSET(4, &s));
  return 0;
}
