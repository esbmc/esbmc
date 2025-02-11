#include <cassert>

struct a;
struct b;
struct c
{
  void d(b);
};
struct a : c
{
  virtual ~a();
};
struct b : a
{
};
