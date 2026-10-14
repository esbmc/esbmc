#include <assert.h>

typedef struct
{
  int anon_pad$1;
  void *p;
} S;

int main()
{
  S s;
  assert(s.anon_pad$1 == 0);
}
