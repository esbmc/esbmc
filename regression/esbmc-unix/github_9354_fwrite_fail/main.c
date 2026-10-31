/* #9354: fwrite reads size * nitems bytes from its buffer. */
#include <stdio.h>
#include <string.h>

int main(void)
{
  char reply[4];
  memcpy(reply, "pong", 4);
  volatile size_t len = 64;
  fwrite(reply, 1, len, stdout);
  return 0;
}
