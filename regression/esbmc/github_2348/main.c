#include <assert.h>

int main(int argc, char *argv[])
{
  if (argc > 10000)
    assert(argv[1000]);
  return 0;
}
