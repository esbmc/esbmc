#include <math.h>

extern void __VERIFIER_error() __attribute__ ((__noreturn__));
void __VERIFIER_assert(int cond) { if (!(cond)) { ERROR: __VERIFIER_error(); } return; }

double cos_pi_x_naive(double x)
{
    double pi = acos(-1);
    return cos(pi * x);
}
// the period is 2, values are (0;0.5) positive, (0.5;1.5) negative, (1.5,2) positive
double cos_pi_x_smart(double x)
{
    int quadrant;
    double rem = remquo(x, 1, &quadrant);
    quadrant = (unsigned)quadrant % 4; // keep 2 bits to determine quadrant
 
    double pi = acos(-1);
    switch(quadrant) {
        case 0: return cos(pi * rem);
        case 1: return -cos(pi * rem);
        case 2: return -cos(pi * rem);
        case 3: return cos(pi * rem);
    };
}

int main(void)
{
  __VERIFIER_assert(0x1.6a09e667f3bcdp-1 == cos_pi_x_naive(0.25));
  __VERIFIER_assert(-0x1.6a09e667f3bcep-1 == cos_pi_x_naive(1.25));
  __VERIFIER_assert(0x1.6a0c104a94705p-1 == cos_pi_x_naive(1000000000000.25));
  __VERIFIER_assert(-0x1.6a0b3ceadf5f9p-1 == cos_pi_x_naive(1000000000001.25));
  __VERIFIER_assert(0x1.6a09e667f3bcdp-1 == cos_pi_x_smart(1000000000000.25));
  __VERIFIER_assert(-0x1.6a09e667f3bcdp-1 == cos_pi_x_smart(1000000000001.25));

  int quo;
  __VERIFIER_assert(isnan(remquo(INFINITY, 1, &quo)));
}
