// Anti-vacuity twin: &pp[1][2] points into r1, not r0, so an equality against
// &r0[2] has to be refuted.
#include <assert.h>

int main(void)
{
  int r0[3] = {0, 1, 2};
  int r1[3] = {3, 4, 5};
  int *rows[2] = {r0, r1};
  int **pp = rows;

  assert(&pp[1][2] == &r0[2]);
  return 0;
}
