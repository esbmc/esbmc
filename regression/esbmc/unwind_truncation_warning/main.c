// --unwind below the trip count with unwinding assertions off assumes the
// rest of the loop away, so a false claim past the bound is discharged in
// silence. The verdict is still SUCCESSFUL; the warning is what makes it
// readable.
#include <assert.h>

int main(void)
{
  int s = 0;
  for (int i = 0; i < 10; i++)
    s++;
  assert(s == 999);
  return 0;
}
