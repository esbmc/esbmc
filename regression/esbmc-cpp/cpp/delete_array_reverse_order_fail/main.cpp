#include <cassert>

int order[7];
int k;

struct E
{
  int id;
  ~E()
  {
    order[k++] = id;
  }
};

int main()
{
  E *p = new E[3];
  for (int i = 0; i < 3; ++i)
    p[i].id = i;
  delete[] p;

  E(*q)[2] = new E[2][2];
  for (int i = 0; i < 4; ++i)
    q[i / 2][i % 2].id = 3 + i;
  delete[] q;
  assert(order[3] == 3);
}
