// github #7548: getline was the last POSIX name the bundled <stdio.h> still
// omitted, so <cstdio> rejected it outright in C++. Pins both the declaration
// (with the ssize_t it needs) and what the model guarantees on success.
#include <cstdio>
#include <cstdlib>
#include <cassert>

int main()
{
  char *line = nullptr;
  size_t cap = 0;

  ssize_t len = getline(&line, &cap, stdin);
  if (len >= 0)
  {
    assert(line != nullptr);
    assert(cap > (size_t)len);
    assert(line[len] == '\0');
  }

  free(line);
  return 0;
}
