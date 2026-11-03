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
  assert(order[0] == 2 && order[1] == 1 && order[2] == 0);

  E(*q)[2] = new E[2][2];
  for (int i = 0; i < 4; ++i)
    q[i / 2][i % 2].id = 3 + i;
  delete[] q;
  assert(order[3] == 6 && order[6] == 3);
}
