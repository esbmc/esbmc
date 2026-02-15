/* Test #pragma unroll N with nested for loops followed by a do-while loop.
 * Uses macro-defined unroll counts and comments between pragmas and loops
 * to reproduce a reported client crash in EvaluateAsInt.
 *
 * Structure:
 *   - Outer for loop (bound 10, pragma unroll OUTER_UNROLL=4)
 *     - Inner for loop (bound 8, pragma unroll INNER_UNROLL=3)
 *   - Do-while loop (bound 10, pragma unroll DOWHILE_UNROLL=5)
 *
 * The matrix[][] accumulation uses values that are safe within the
 * unrolled iterations, and the do-while loop performs a post-processing
 * reduction. Assertions verify correctness under the pragma limits.
 */

#include <assert.h>
#include <stdint.h>

#define OUTER_UNROLL 4
#define INNER_UNROLL 3
#define DOWHILE_UNROLL 5

int main()
{
  int matrix[4][3] = {{0}};
  int row_sum[4] = {0};

  /* Nested loops: outer unrolled 4 times, inner unrolled 3 times.
   * Only iterations i in [0,3] and j in [0,2] are explored. */
  #pragma unroll OUTER_UNROLL
  /* Populate the matrix row by row */
  for (uint32_t i = 0; i < 10; i++)
  {
    #pragma unroll INNER_UNROLL
    /* Fill each column in the current row */
    for (uint32_t j = 0; j < 8; j++)
    {
      matrix[i][j] = (i + 1) * (j + 1);
    }
  }

  /* Verify the nested loop results for the explored iterations. */
  assert(matrix[0][0] == 1);  /* (0+1)*(0+1) */
  assert(matrix[0][2] == 3);  /* (0+1)*(2+1) */
  assert(matrix[2][1] == 6);  /* (2+1)*(1+1) */
  assert(matrix[3][2] == 12); /* (3+1)*(2+1) */

  /* Do-while loop: unrolled 5 times, but only indices 0..3 are valid rows.
   * Accumulates row sums from the matrix. */
  int k = 0;
  #pragma unroll DOWHILE_UNROLL
  /* Reduce each row into a scalar sum */
  do
  {
    if (k < 4)
    {
      for (int c = 0; c < 3; c++)
        row_sum[k] += matrix[k][c];
    }
    k++;
  } while (k < 10);

  /* Verify the do-while loop accumulated correct row sums.
   * row 0: 1*1 + 1*2 + 1*3 = 6
   * row 1: 2*1 + 2*2 + 2*3 = 12
   * row 2: 3*1 + 3*2 + 3*3 = 18
   * row 3: 4*1 + 4*2 + 4*3 = 24
   */
  assert(row_sum[0] == 6);
  assert(row_sum[1] == 12);
  assert(row_sum[2] == 18);
  assert(row_sum[3] == 24);

  return 0;
}
