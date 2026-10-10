#include <stdlib.h>

struct player
{
  char name[20];
  int score;
};

static int compare(struct player a, struct player b)
{
  return b.score - a.score;
}

int main(void)
{
  struct player p[2] = {{"a", 1}, {"b", 2}};
  qsort(p, 2, sizeof p[0], (void *)compare);
  return 0;
}
