#include <assert.h>

/* The mirror of github_1932: a cast to an integer narrower than a pointer
 * loses the high bits, so a zero result does not imply a null pointer and the
 * assertion must not be provable. Pins that the pointer/integer round-trip is
 * not treated as lossless regardless of the cast's width. */

int main(int argc, char **argv) {
  if ((unsigned int)argv[0] == (unsigned int)0) {
    assert(argv[0] == 0);
  }
  return 0;
}
