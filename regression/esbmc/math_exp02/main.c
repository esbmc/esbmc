#include <math.h>
#include <stdio.h>
#include <assert.h>

#define M_PI 3.14159265358979323846
#define PREC 1e-16
#define M_LN10   2.30258509299404568402

double inline fabs(double x) {
  return (x < 0) ? -x : x;
}

#define DBL_EPSILON 2.2204460492503131e-16


#define M_E     2.71828182845905
#define M_E2    (M_E * M_E)
#define M_E4    (M_E2 * M_E2)
#define M_E8    (M_E4 * M_E4)
#define M_E16   (M_E8 * M_E8)
#define M_E32   (M_E16 * M_E16)
#define M_E64   (M_E32 * M_E32)
#define M_E128  (M_E64 * M_E64)
#define M_E256  (M_E128 * M_E128)
#define M_E512  (M_E256 * M_E256)
#define M_E1024 (M_E512 * M_E512)

static double _expi_square_tbl[11] = {
        M_E,            // e^1
        M_E2,           // e^2
        M_E4,           // e^4
        M_E8,           // e^8
        M_E16,          // e^16
        M_E32,          // e^32
        M_E64,          // e^64
        M_E128,         // e^128
        M_E256,         // e^256
        M_E512,         // e^512
        M_E1024,        // e^1024
};

static double _expi(int n) {
        int i;
        double val;

        if (n > 1024) {
//                return FP_INFINITE;
            return (1.0/0.0);
        }

        val = 1.0;

        for (i = 0; n; i++) {
                if (n & (1 << i)) {
                        n &= ~(1 << i);
                        val *= _expi_square_tbl[i];
                }
        }

        return val;
}

static double _dbl_inv_fact[] = {
        1.0 / 1.0,                                      // 1 / 0!
        1.0 / 1.0,                                      // 1 / 1!
        1.0 / 2.0,                                      // 1 / 2!
        1.0 / 6.0,                                      // 1 / 3!
        1.0 / 24.0,                                     // 1 / 4!
        1.0 / 120.0,                            // 1 / 5!
        1.0 / 720.0,                            // 1 / 6!
        1.0 / 5040.0,                           // 1 / 7!
        1.0 / 40320.0,                          // 1 / 8!
        1.0 / 362880.0,                         // 1 / 9!
        1.0 / 3628800.0,                        // 1 / 10!
        1.0 / 39916800.0,                       // 1 / 11!
        1.0 / 479001600.0,                      // 1 / 12!
        1.0 / 6227020800.0,                     // 1 / 13!
        1.0 / 87178291200.0,            // 1 / 14!
        1.0 / 1307674368000.0,          // 1 / 15!
        1.0 / 20922789888000.0,         // 1 / 16!
        1.0 / 355687428096000.0,        // 1 / 17!
        1.0 / 6402373705728000.0,       // 1 / 18!
};

int main() {
  printf("%.16f \n", exp(0.3));
  printf("%.16f \n", exp(0.14115125));
  printf("%.16f \n", exp(-2.132314121515));
  printf("%.16f \n", exp(3.1123441));
  printf("%.16f \n", exp(-10.34));

  assert(fabs(exp(0.3) - 1.3498588075760032) <= 1e-8 );
  assert(fabs(exp(0.14115125) - 1.1515988141337348) <= 1e-9 );
  assert(fabs(exp(-2.132314121515) - 0.11856260786488046) <= 1e-10 );
  assert(fabs(exp(3.1123441) - 22.4736632187176717) <= 1e-7);
  assert(fabs(exp(-10.34) - 3.231432266044366e-05) <= 1e-10);
  return 0;
}

