//#include <assert.h>

void func(int a[], int n)
{
  assert(a[1] == 1);
}

int main()
{
  int a[2] = {0, 1};
  func(a, 2);
}
