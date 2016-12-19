#include <stdio.h>
#include <math.h>
//#include <fenv.h>
//#include <float.h>
#include <assert.h>

extern void __VERIFIER_error() { assert(0); };
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

float my_nextafterf( float from, float to )
{
  int from_int = *(unsigned long int*)(&from);

  if(from < to)
    from_int++;
  else
    from_int--;

  return *(float*)(&from_int);
}

int main(void)
{
//    float from1 = 0, to1 = nextafterf(from1, 1);
//    assert(to1 == 0x1p-149);
 
    float from2 = 1;
//    float to21 = nextafterf(from2, 2);
//    printf("%.64f (%a)\n", to21, to21);
//    assert(to21 == 0x1.000002p+0);

    float to22 = my_nextafterf(from2, 2);
    printf("%.64f (%a)\n", to22, to22);
    assert(to22 == 0x1.000002p+0);

//    float inc = FLT_MIN * FLT_EPSILON;
//    float a = 1.0 + inc;
//    
//    x++;
//    a = *(float*)(&x);
//    printf("%.20f (%a)\n", a, a);
//    assert(0 == a);
/*
 
    double from3 = nextafter(0.1, 0), to3 = 0.1;
    printf("The number 0.1 lies between two valid doubles:\n"
           "    %.56f (%a)\nand %.55f  (%a)\n", from3, from3, to3, to3);
 
    // difference between nextafter and nexttoward:
    long double dir = nextafterl(from1, 1); // first subnormal long double
    float x = nextafterf(from1, dir); // first converts dir to float, giving 0
    printf("Using nextafter, next float after %.2f (%a) is %.20g (%a)\n",
           from1, from1, x, x);
    x = nexttowardf(from1, dir);
    printf("Using nexttoward, next float after %.2f (%a) is %.20g (%a)\n",
           from1, from1, x, x);
 
    double from4 = DBL_MAX, to4 = nextafter(from4, INFINITY);
    printf("The next representable double after %.2g (%a) is %.23f (%a)\n",
               from4, from4, to4, to4);

    float from5 = 0.0, to5 = nextafter(from5, -0.0);
    printf("nextafter(+0.0, -0.0) gives %.2g (%a)\n", to5, to5);

    printf ("first representable value greater than zero: %e\n", nextafter(0.0,1.0));
    printf ("first representable value less than zero: %e\n", nextafter(0.0,-1.0));
*/
}
