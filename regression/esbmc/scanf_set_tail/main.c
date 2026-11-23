#include <assert.h>
#include <stdio.h>

struct rec
{
  int id;
  char name[8];
  int tag;
} r = {7, "ab", 9};

int main()
{
  sscanf("xyz", "%[a-z]", &r.name[1]);
  assert(r.id == 7);
  assert(r.name[0] == 'a');
  return 0;
}
