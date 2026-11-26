/* Each write extends the `with` chain carried for `a`. Re-checking the whole
 * chain at every write and read made symex quadratic: minutes, not seconds. */
#include <assert.h>

#define N 16000
int a[N];

int main()
{
  for (int i = 0; i < N; ++i)
    a[i] = i;
  for (int i = 0; i < N; ++i)
    assert(a[i] == i);
  return 0;
}
