// &f[i] through a user operator[] returning T& points into the array.
#include <cassert>

struct Cols
{
  unsigned short *ptr_;
  unsigned short &operator[](long i) const
  {
    return ptr_[i];
  }
};

int main()
{
  unsigned short frac[8] = {0};
  Cols f{frac};
  unsigned short *q = &f[0];
  q[7] = 5;
  assert(frac[7] == 5);
  assert(&f[3] == frac + 3);
  return q[7];
}
