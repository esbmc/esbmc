// Negative counterpart of github_7643_complex_init: the second initialiser is
// the imaginary part, not the real one.
#include <assert.h>

int main(void)
{
  _Complex int z = {1, 2};
  assert(__real__ z == 2);
  return 0;
}
