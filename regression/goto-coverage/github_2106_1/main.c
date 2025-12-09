/* MDH WCET BENCHMARK SUITE. File version $Id: expint.c,v 1.3 2005/11/11 10:29:56 ael01 Exp $ */

/************************************************************************
 * FROM:
 *   http://sron9907.sron.nl/manual/numrecip/c/expint.c
 *
 * FEATURE:
 *   One loop depends on a loop-invariant value to determine
 *   if it run or not.
 *
 ***********************************************************************/

 /*
  * Changes: JG 2005/12/23: Changed type of main to int, added prototypes.
                            Indented program.
  */

long int        foo(long int x);
long int        expint(int n, long int x);

int
main()
{
    expint(50, 1);
    /* with  expint(50,21) as argument, runs the short path */
    /* in expint.   expint(50,1)  gives the longest execution time */
    return 0;
}

long int
foo(long int x)
{
    return (x * x + (8 * x)) << (4 - x);
}


/* Function with same flow, different data types,
   nonsensical calculations */
long int
expint(int n, long int x)
{
    int             i, ii, nm1;
    long int        a, b, c, d, del, fact, h, psi, ans;

    nm1 = n - 1;        /* arg=50 --> 49 */

    if (x > 1) {        /* take this leg? */
        b = x + n;
        c = 2e6;
        d = 3e7;
        h = d;

        for (i = 1; i <= 100; i++) {    /* MAXIT is 100 */
            a = -i * (nm1 + i);
            b += 2;
            d = 10 * (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (del < 10000) {
                ans = h * -x;
                return ans;
            }
        }
    } else {        /* or this leg? */
        /* For the current argument, will always take */
        /* '2' path here: */
        ans =  1000;
        fact = 1;
        for (i = 1; i <= 100; i++) {    /* MAXIT */
            fact *= -x / i;
            if (i != nm1)    /* depends on parameter n */
                del = -fact / (i - nm1);
            else {    /* this fat piece only runs ONCE *//* runs on
                 * iter 49 */
                psi = 0x00FF;
                for (ii = 1; ii <= nm1; ii++)    /* */
                    psi += ii + nm1;
                del = psi + fact * foo(x);
            }
            ans += del;
            /* conditional leave removed */
        }

    }
    return ans;
}
