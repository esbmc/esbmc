#include <assert.h>
#include <stdlib.h>

struct V
{
  int *buf;
  int size;
};

int main(void)
{
  struct V s;
  s.buf = malloc(2 * sizeof(int));
  s.size = 2;
  struct V out = s;
  struct V *q = &out;
  int n = 0;
  for (int i = 0; i < out.size; i++)
    n++;
  assert(n == 2 && q->buf == s.buf);
  return 0;
}
