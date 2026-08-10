// Negative counterpart of wchar_model: the models really write, so a claim
// contradicting the copy is refuted rather than vacuously held. An unmodelled
// wcscpy -- one that returned nondet and wrote nothing -- would let this pass
// (issue #5868).
#include <wchar.h>

int main(void)
{
  wchar_t src[4] = {'a', 'b', 'c', 0};
  wchar_t dst[4] = {'z', 'z', 'z', 'z'};
  wcscpy(dst, src);
  __ESBMC_assert(dst[0] == 'z', "must not hold");
  return 0;
}
