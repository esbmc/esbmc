#include<cassert>

void f(int &x) try {
  throw 10;
}
catch (const int &i)
{
  x = i;
}

int main() {
  int v = 0;
  f(v);
  assert(v==10);
}

