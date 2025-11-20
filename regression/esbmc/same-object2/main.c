#include <assert.h>
#include <stdlib.h>

// Existing structs
struct Point {
    int x, y;
};

struct Container {
    int data[3];
    int value;
};

int main()
{
  int sx, sy;
  int *sp, *sq;

  // Pointers may or may not alias → cannot be simplified
  // These assertions should remain satisfiable in both directions
  assert(__ESBMC_same_object(sp, sp));     // reflexivity must hold
  assert(__ESBMC_same_object(sq, sq));

  // Negated case should not trivially hold (let solver decide)
  // For example, sp and sq *could* point to the same object
  __ESBMC_assume(sp == &sx || sp == &sy);
  __ESBMC_assume(sq == &sx || sq == &sy);
  // At this point, __ESBMC_same_object(sp, sq) is solver-dependent

  int arr[4];
  int idx1, idx2;
  __ESBMC_assume(idx1 >= 0 && idx1 < 4);
  __ESBMC_assume(idx2 >= 0 && idx2 < 4);

  // Same array, symbolic indices → must be same object
  assert(__ESBMC_same_object(&arr[idx1], &arr[idx2]));

  // Different array objects → must be false
  int other[4];
  assert(!__ESBMC_same_object(&arr[idx1], &other[idx2]));

  struct Point spoint1, spoint2;
  int choose;

  // Choose either x or y field of spoint1
  int *field1 = (choose ? &spoint1.x : &spoint1.y);

  // Same struct, different field pointer → must be same object
  assert(__ESBMC_same_object(&spoint1.x, field1));

  // Different structs → must be false
  assert(!__ESBMC_same_object(&spoint1.x, &spoint2.x));

  return 0;
}


