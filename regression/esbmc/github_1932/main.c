// The round trip has to be through a pointer-sized integer: `unsigned long`
// is 32-bit under LLP64, so the cast truncated argv[0] and the guard admitted
// non-null pointers whose low word happened to be zero. #1932
#include <assert.h>
#include <stdint.h>

int main(int argc, char **argv) {
  if ((uintptr_t)argv[0] == (uintptr_t)0) {
    assert(argv[0] == 0);
  }
  return 0;
}
