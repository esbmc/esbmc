#define TRUE 1
#define FALSE 0

float shiftL(float zIn, float z[], int N)
{
	int i;
	float zOut;
	zOut = z[0];
	for (i=0; i<=N-2; i++){
		z[i] = z[i+1];
	}
	z[N-1] = zIn;
	return (zOut);
}

int order(int Na, int Nb)
{
	return Na>Nb ? Na-1 : Nb-1;
}

float powInt(int x, int y)
{
	int i;
	float ret = 1.0;
	for(i=0; i<y; ++i)
	{
		ret *= x;
	}
	return ret;
}

int roundInt(float a) {
  if (a>=0)
    return (int)(a+0.5);
  else
    return (int)(a-0.5);
}

float maxFixed(int k, int l)
{
	return (float)powInt(2,k-1)-(1.0/powInt(2,l));
}

float minFixed(int k, int l)
{
	return (-1.0)*(float)powInt(2,k-1);
}

float wrapF(float fX, float fLowerBound, float fUpperBound)
{
	int kX = (int)fX;
	int kLowerBound = (int)fLowerBound;
	int kUpperBound = (int)fUpperBound;

	float f = (fX >= 0) ? fX-kX : kX-fX;

	int range_size = kUpperBound - kLowerBound + 1;

    if (kX < kLowerBound)
        kX += range_size * ((kLowerBound - kX) / range_size + 1);

    int ret = (kLowerBound + (kX - kLowerBound) % range_size);
    return (ret >= 0) ? ret + f : ret - f;
}

float floatToFix(float number, float delta, int k, int l, int catchOverflow)
{
	//float delta = 1.0/powInt(2,l);
	float div = roundInt(number/delta);
	float ret = delta*div;
	if (catchOverflow)
	{
		assert(ret <= maxFixed(k,l) && ret >= minFixed(k,l));
	}
	else if (ret > maxFixed(k,l) || ret < minFixed(k,l))
	{
		ret = wrapF(ret, minFixed(k,l), maxFixed(k,l)); //wrap around
	}
	return ret;
}

float fixedDelta(int l)
{
	return 1.0/powInt(2,l);
}

float iirOutFixed(float y[], float x[], float a[], float b[], int Na, int Nb, int k, int l, float delta, int catchOverflow)
{																			// timer1 += 8;
	float *a_ptr, *y_ptr, *b_ptr, *x_ptr;									// timer1 += 4;
	float sum = 0;															// timer1 += 5;
	a_ptr = &a[1];															// timer1 += 1;
	y_ptr = &y[Na-1];														// timer1 += 5;
	b_ptr = &b[0];															// timer1 += 1;
	x_ptr = &x[Nb-1];														// timer1 += 6;
	int i, j;
	for (i = 0; i < Nb; i++){												// timer1 += 3;
		sum += *b_ptr++ * *x_ptr--;											// timer1 += (5+3+3+3);
		sum = floatToFix(sum, delta, k, l, catchOverflow);
	}
	for (j = 0; j < Na-1; j++){												// timer1 += 2;
		sum -= *a_ptr++ * *y_ptr--;											// timer1 += (5+5+3+1);
		sum = floatToFix(sum, delta, k, l, catchOverflow);
	}																		// timer1 += 4;
	return sum;																// timer1 += 4;
}																			// timer1 += 3;


int nondet_int();
float nondet_float();

int main(void)
{
	int i;
	int xsize = 6; 	
	int Na = 2;	
	int Nb = 1;	
	int k = 2; 	
	int l = 4; 	
	float max = 1.0;
	float min = -1.0;
	float delta = fixedDelta(l);
	assert(delta>0.0);

	float a[Na];
	float b[Nb];


	a[0] = 1.0000; a[1] = -0.5;
	b[0] = 1.0000;

	for (i=0; i<Na; ++i)
	{
		a[i] = floatToFix(a[i], delta, k, l, TRUE);
	}
	for (i=0; i<Nb; ++i)
	{
		b[i] = floatToFix(b[i], delta, k, l, TRUE);
	}

	float y[xsize];
	float x[xsize];
	for (i = 0; i<xsize; ++i)
	{
		y[i] = 0;
		x[i] = nondet_int()*delta; //nondet_float();
		__ESBMC_assume(x[i]>=min && x[i]<=max);
	}

	float yaux[Na];
	float xaux[Nb];
	for (i = 0; i<Na; ++i)
	{
		yaux[i] = 0;
	}
	for (i = 0; i<Nb; ++i)
	{
		xaux[i] = 0;
	}

	for(i=0; i<xsize; ++i)
	{
		shiftL(x[i],xaux,Nb);
		y[i] = iirOutFixed(yaux,xaux,a,b,Na,Nb,k,l, delta, TRUE);
		shiftL(y[i],yaux,Na);
	}

	return 0;
}

