#include <stdio.h>
#include <stdint.h>
#include <assert.h>

union cheri
{
  void *cap;
  struct
  {
    uint64_t c;
    uint64_t p;
  };
};

int main()
{
  char *cap_ptr = "hello"; // 64bits
  union cheri u = {cap_ptr};
  u.p = 0xFFFFFFFFFFFFFFFF;
  char *initial_cursor = u.cap;

  for (int i = 0; i < 6; i++)
  {
    assert(initial_cursor[i] == cap_ptr[i]);
  }

  return 0;
}
