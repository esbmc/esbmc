#include <assert.h>

int main(int argc, char **argv) {
  if ((unsigned long)argv[0] == (unsigned long)0) {
    assert(argv[0] == 0);
  }
  return 0;
}

