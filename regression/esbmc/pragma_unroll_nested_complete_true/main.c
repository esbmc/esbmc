/* Test #pragma unroll N with nested loops where inner loop completes
 * Outer loop has bound 10, pragma limits to 5
 * Inner loop has bound 2, pragma limits to 3 (loop completes in 2)
 * This verifies outer loop pragma is applied when inner loop finishes
 */

#include <stdint.h>

int main() {
  int sum = 0;

  #pragma unroll 5
  for (uint32_t i = 0; i < 10; i++) {

    #pragma unroll 3
    for (uint32_t j = 0; j < 2; j++) {
      sum += i + j;
    }
  }

  // Verification succeeds with limited unrolling
  return 0;
}
