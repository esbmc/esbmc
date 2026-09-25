#include <assert.h>
#include <stdint.h>

/* The issue's reproducer casts through `unsigned long`, which is pointer-sized
 * only on LP64. On LLP64 (Windows) it is 32 bits, so the guard truncates and
 * does not imply a null pointer -- see github_1932_lossy_cast_fail. `uintptr_t`
 * is pointer-sized on every target, which is what the reproducer meant. */

int main(int argc, char **argv) {
  if ((uintptr_t)argv[0] == (uintptr_t)0) {
    assert(argv[0] == 0);
  }
  return 0;
}
