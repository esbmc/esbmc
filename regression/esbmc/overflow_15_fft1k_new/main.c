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
/*  FILE: fft1k.c                                                        */
/*  SOURCE : C Algorithms for Real-Time DSP by P. M. Embree              */
/*                                                                       */
/*  DESCRIPTION :                                                        */
/*                                                                       */
/*     FFT (Fast Fourier Transform) for complex number arrays.           */
/*     The input complex number array is w[] and the result is stored in */
/*     the array x[].                                                    */
/*     The array w[] is initialized in the function init_w.              */
/*                                                                       */
/*  REMARK :                                                             */
/*                                                                       */
/*  EXECUTION TIME :                                                     */
/*                                                                       */
/*                                                                       */
/*************************************************************************/



typedef struct {
    float real, imag;
} COMPLEX;

void fft_c(int n);
void init_w(int n);

#define PI 3.14159265358979323846
int n = 1024;
COMPLEX x[1024],w[1024];


float fabs(float n)
{
  float f;

  if (n >= 0) f = n;
  else f = -n;
  return f;
}


float sin(float rad)
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

float cos(float rad)
//float rad;
{
//  float sin();

  return (sin (PI / 2.0 - rad));
}

    

void main()
{
    int i;

    init_w(n);

    x[0].real = 1.0;
    fft_c(n);
}

void fft_c(int n)
{
    COMPLEX u,temp,tm;
    COMPLEX *xi,*xip,*wptr;

    int i,j,le,windex;

/* start fft */

    windex = 1;
    for(le=n/2 ; le > 0 ; le/=2) {
        wptr = w;
        for (j = 0 ; j < le ; j++) {
            u = *wptr;
            for (i = j ; i < n ; i = i + 2*le) {
                xi = x + i;
		xip = xi + le;
                temp.real = xi->real + xip->real;
                temp.imag = xi->imag + xip->imag;
                tm.real = xi->real - xip->real;
                tm.imag = xi->imag - xip->imag;             
                xip->real = tm.real*u.real - tm.imag*u.imag;
                xip->imag = tm.real*u.imag + tm.imag*u.real;
                *xi = temp;
            }
            wptr = wptr + windex;
        }
        windex = 2*windex;
    }            
}
void init_w(int n)
{
    int i;
    float a = 2.0*PI/n;
    for(i = 0 ; i < n ; i++) {
        w[i].real = cos(i*a);
        w[i].imag = sin(i*a);
    }
}
