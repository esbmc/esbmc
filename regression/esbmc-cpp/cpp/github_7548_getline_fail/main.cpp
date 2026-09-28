// github #7548: the negative twin. getline terminates the buffer it returns,
// so the byte at the returned length is never anything but '\0'.
#include <cstdio>
#include <cstdlib>
#include <cassert>

int main()
{
  char *line = nullptr;
  size_t cap = 0;

  ssize_t len = getline(&line, &cap, stdin);
  if (len >= 0)
    assert(line[len] != '\0');

  free(line);
  return 0;
}
