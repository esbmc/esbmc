#include <stdlib.h>

#ifdef _WIN32
#define random() rand()
#endif

#include "ieee_float.h"

float random_float()
{
  union
  {
    float f;
    unsigned i;
  } u;

  u.i=random();
  u.i=(u.i<<16)^random();

  return u.f;
}

bool eq(const ieee_floatt &a, const ieee_floatt &b)
{
  if(a.is_NaN() && b.is_NaN()) return true;
  if(a.is_infinity() && b.is_infinity() && a.get_sign()==b.get_sign()) return true;
  return a==b;
}

typedef enum { PLUS=0, MINUS=1, MULT=2, DIV=3 } binopt;
const char *binopsyms[]={ " + ", " - ", " * ", " / " };

int main()
{
  ieee_floatt i1, i2, i3, res;
  float f1, f2, f3;

  for(unsigned i=0; i<100000000; i++)
  {
    if(i%100000==0) std::cout << "*********** " << i << std::endl;

    f1=random_float();
    f2=random_float();
    i1.from_float(f1);
    i2.from_float(f2);
    res=i1;
    f3=f1;

    int op=(binopt)i%4;

    switch(op)
    {
    case PLUS:
      f3+=f2;
      res+=i2;
      break;

    case MINUS:
      f3-=f2;
      res-=i2;
      break;

    case MULT:
      f3*=f2;
      res*=i2;
      break;

    case DIV:
      f3/=f2;
      res/=i2;
      break;

    default:assert(0);
    }

    i3.from_float(f3);

    if(!eq(res, i3))
    {
      const char *opsym=binopsyms[op];
      std::cout << i1 << opsym << i2 << " != " << res << std::endl;
      std::cout << f1 << opsym << f2 << " == " << f3 << std::endl;
      std::cout << integer2binary(i1.get_fraction(), i1.spec.f+1) << opsym <<
                   integer2binary(i2.get_fraction(), i1.spec.f+1) << " != " <<
                   integer2binary(res.get_fraction(), i1.spec.f+1) <<
                   " (" << res.get_fraction() << ")" << std::endl;
      std::cout << integer2binary(i1.get_fraction(), i1.spec.f+1) << opsym <<
                   integer2binary(i2.get_fraction(), i1.spec.f+1) << " == " <<
                   integer2binary(i3.get_fraction(), i1.spec.f+1) <<
                   " (" << i3.get_fraction() << ")" << std::endl;
      std::cout << std::endl;
    }
  }
}
