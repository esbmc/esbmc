/*************************************************************************/
/*                                                                       */
/*   SNU-RT Benchmark Suite for Worst Case Timing Analysis               */
/*   =====================================================               */
/*                              Collected and Modified by S.-S. Lim      */
/*                                           sslim@archi.snu.ac.kr       */
/*                                         Real-Time Research Group      */
/*                                        Seoul National University      */
/*                                                                       */
/*                                                                       */
/*        < Features > - restrictions for our experimental environment   */
/*                                                                       */
/*          1. Completely structured.                                    */
/*               - There are no unconditional jumps.                     */
/*               - There are no exit from loop bodies.                   */
/*                 (There are no 'break' or 'return' in loop bodies)     */
/*          2. No 'switch' statements.                                   */
/*          3. No 'do..while' statements.                                */
/*          4. Expressions are restricted.                               */
/*               - There are no multiple expressions joined by 'or',     */
/*                'and' operations.                                      */
/*          5. No library calls.                                         */
/*               - All the functions needed are implemented in the       */
/*                 source file.                                          */
/*                                                                       */
/*                                                                       */
/*************************************************************************/
/*                                                                       */
/*  FILE: fir.c                                                          */
/*  SOURCE : C Algorithms for Real-Time DSP by P. M. Embree              */
/*                                                                       */
/*  DESCRIPTION :                                                        */
/*                                                                       */
/*     An example using FIR filter and Gaussian function.                */
/*     algorithm.                                                        */
/*     The function 'fir_filter' is for FIR filtering and the function   */
/*     'gaussian' is for Gaussian number generation.                     */
/*     The detailed description is above each function.                  */
/*                                                                       */
/*  REMARK :                                                             */
/*                                                                       */
/*  EXECUTION TIME :                                                     */
/*                                                                       */
/*                                                                       */
/*************************************************************************/


#define SAMPLE_RATE 11025
#define RAND_MAX 32768
#define PI 3.14159265358979323846



/* function prototypes for fft and filter functions */
float fir_filter(float input,float *coef,int n,float *history);

static float gaussian(void);


/* FILTER COEFFECIENTS FOR FILTER ROUTINES */

/* FILTERS: 2 FIR AND 2 IIR */

/* 35 point lowpass FIR filter cutoff at 0.19
 designed using the Parks-McClellan program */

  float  fir_lpf35[35] = {
  -6.3600959e-03,  -7.6626200e-05,   7.6912856e-03,   5.0564148e-03,  -8.3598122e-03,
  -1.0400905e-02,   8.6960020e-03,   2.0170502e-02,  -2.7560785e-03,  -3.0034777e-02,
  -8.9075034e-03,   4.1715767e-02,   3.4108155e-02,  -5.0732918e-02,  -8.6097546e-02,
   5.7914939e-02,   3.1170085e-01,   4.4029310e-01,   3.1170085e-01,   5.7914939e-02,
  -8.6097546e-02,  -5.0732918e-02,   3.4108155e-02,   4.1715767e-02,  -8.9075034e-03,
  -3.0034777e-02,  -2.7560785e-03,   2.0170502e-02,   8.6960020e-03,  -1.0400905e-02,
  -8.3598122e-03,   5.0564148e-03,   7.6912856e-03,  -7.6626200e-05,  -6.3600959e-03
                          };

/* 37 point lowpass FIR filter cutoff at 0.19
 designed using the KSRFIR.C program */

  float  fir_lpf37[37] = {
  -6.51000e-04,  -3.69500e-03,  -6.28000e-04,   6.25500e-03,   4.06300e-03,
  -8.18900e-03,  -1.01860e-02,   7.84700e-03,   1.89680e-02,  -3.05100e-03,
  -2.96620e-02,  -9.06500e-03,   4.08590e-02,   3.34840e-02,  -5.07550e-02,
  -8.61070e-02,   5.75690e-02,   3.11305e-01,   4.40000e-01,   3.11305e-01,
   5.75690e-02,  -8.61070e-02,  -5.07550e-02,   3.34840e-02,   4.08590e-02,
  -9.06500e-03,  -2.96620e-02,  -3.05100e-03,   1.89680e-02,   7.84700e-03,
  -1.01860e-02,  -8.18900e-03,   4.06300e-03,   6.25500e-03,  -6.28000e-04,
  -3.69500e-03,  -6.51000e-04
                          };


/**************************************************************************

fir_filter - Perform fir filtering sample by sample on floats

Requires array of filter coefficients and pointer to history.
Returns one output sample for each input sample.

float fir_filter(float input,float *coef,int n,float *history)

    float input        new float input sample
    float *coef        pointer to filter coefficients
    int n              number of coefficients in filter
    float *history     history array pointer

Returns float value giving the current output.

*************************************************************************/

int Cnt1, Cnt2, Cnt3, Cnt4;


int rand()
{
  static unsigned long next = 1;

  next = next * 1103515245 + 12345;
  return (unsigned int)(next/65536) % 32768;
}


static float fabs(float n)
{
  float f;

  if (n >= 0) f = n;
  else f = -n;
  return f;
}


static float sin(float rad)
//float rad;
{
  float app;

  float diff;
  int inc = 1;

  while (rad > 2*PI)
	rad -= 2*PI;
  while (rad < -2*PI)
    rad += 2*PI;
  app = diff = rad;
   diff = (diff * (-(rad*rad))) /
      ((2.0 * inc) * (2.0 * inc + 1.0));
    app = app + diff;
    inc++;
  while(fabs(diff) >= 0.00001) {
    diff = (diff * (-(rad*rad))) /
      ((2.0 * inc) * (2.0 * inc + 1.0));
    app = app + diff;
    inc++;
  }

  return(app);
}


static float log(float r)
//float r;
{
  return 4.5;
}


static float sqrt(float val)
//float val;
{
  float x = val/10;

  float dx;

  double diff;
  double min_tol = 0.00001;

  int i, flag;

  flag = 0;
  if (val == 0 ) x = 0;
  else {
    for (i=1;i<20;i++)
      {
	if (!flag) {
	  dx = (val - (x*x)) / (2.0 * x);
	  x = x + dx;
	  diff = val - (x*x);
	  if (fabs(diff) <= min_tol) flag = 1;
	}
	else 
	  x =x;
      }
  }
  return (x);
}


float fir_filter(float input,float *coef,int n,float *history)
{
    int i;
    float *hist_ptr,*hist1_ptr,*coef_ptr;
    float output;

    hist_ptr = history;
    hist1_ptr = hist_ptr;             /* use for history update */
    coef_ptr = coef + n - 1;          /* point to last coef */

/* form output accumulation */
    output = *hist_ptr++ * (*coef_ptr--);
#ifdef DEBUG
    if (n > Cnt2) Cnt2 = n;
#endif
    for(i = 2 ; i < n ; i++) {
        *hist1_ptr++ = *hist_ptr;            /* update history array */
        output += (*hist_ptr++) * (*coef_ptr--);
    }
    output += input * (*coef_ptr);           /* input tap */
    *hist1_ptr = input;                      /* last history */

    return(output);
}


/**************************************************************************

gaussian - generates zero mean unit variance Gaussian random numbers

Returns one zero mean unit variance Gaussian random numbers as a double.
Uses the Box-Muller transformation of two uniform random numbers to
Gaussian random numbers.

float gaussian()

*************************************************************************/

static float gaussian()
{
    static int ready = 0;       /* flag to indicated stored value */
    static float gstore;        /* place to store other value */
    static float rconst1 = (float)(2.0/RAND_MAX);
    static float rconst2 = (float)(RAND_MAX/2.0);
    float v1,v2,r,fac;
    float gaus;

/* make two numbers if none stored */
    if(ready == 0) {
      v1 = (float)rand() - rconst2;
      v2 = (float)rand() - rconst2;
      v1 *= rconst1;
      v2 *= rconst1;
      r = v1*v1 + v2*v2;
      while(r > 1.0f) {
	v1 = (float)rand() - rconst2;
	v2 = (float)rand() - rconst2;
	v1 *= rconst1;
	v2 *= rconst1;
	r = v1*v1 + v2*v2;
#ifdef DEBUG
	printf("*");
#endif
      }        /* make radius less than 1 */

/* remap v1 and v2 to two Gaussian numbers */
      /*        fac = sqrt(-2.0f*log(r)/r);  */
        fac = sqrt(-2.0f * 0.1);
        gstore = v1*fac;        /* store one */
        gaus = v2*fac;          /* return one */
        ready = 1;              /* set ready flag */
    }

    else {
        ready = 0;      /* reset ready flag for next pair */
        gaus = gstore;  /* return the stored one */
    }

    return(gaus);
}


/***********************************************************************

MKGWN.C - Gaussian Noise Filter Example

This program filters a sine wave with added Gaussian noise.  It
implements a 35 point FIR filter (stored in variable fir_lpf35)
on an generated signal.  The filter is a LPF with 40 dB out of
band rejection.  The 3 dB point is at a relative frequency of
approximately .25*fs.

************************************************************************/

float sigma = 0.2;

float nondet_float();

void main()
{
  int          i, j;
  float x;
  static float hist[34];
/* first with filter */
  for(i = 0 ; i < 10 ; i++) {
      x = nondet_float();
      x *= 25000.0;         /* scale for D/A converter */
      fir_filter(x,fir_lpf35,35,hist);
  }
/* now without filter */
  for(i = 0 ; i < 10 ; i++) {
      x = nondet_float();
      x *= 25000.0;         /* scale for D/A converter */
  }

#ifdef DEBUG
	printf("n=%d\n",Cnt2);
#endif
}
