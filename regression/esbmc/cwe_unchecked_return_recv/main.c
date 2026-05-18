#include <sys/socket.h>

int main(void)
{
  char buf[64];
  long n = recv(0, buf, sizeof(buf), 0);
  return (int)(n + 1);
}
