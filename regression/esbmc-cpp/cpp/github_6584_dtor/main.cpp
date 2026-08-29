// Construction without matching destruction would report a spurious leak, so
// pin that delete[] releases what the element constructors acquired.
#include <cstdlib>

struct R
{
  int *p;
  R()
  {
    p = (int *)malloc(sizeof(int));
  }
  ~R()
  {
    free(p);
  }
};

int main()
{
  R *a = new R[3];
  delete[] a;
  return 0;
}
