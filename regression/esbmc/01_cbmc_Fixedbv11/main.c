/**
 * ============================================================================
 * Name        : verification.c
 * Author      : Renato Abreu
 * Version     :
 * Copyright   : Copyright by Renato Abreu
 * Description : BMC verification of FIR/IIR filters
 * ============================================================================
 */

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

int roundInt(float number)
{
	float ret = (number >= 0) ? (number + 0.5) : (number - 0.5);
    return (int)ret;
}

int ceilInt(float number)
{
	float ret = (number - (int)number > 0) ? (number + 1.0) : (number);
    return (int)ret;
}

int floorInt(float number)
{
	float ret = (number - (int)number > 0) ? (number) : (number - 1.0);
    return (int)ret;
}

float maxFixed(int k, int l)
{
	return (float)powInt(2,k-1)-(1.0/powInt(2,l));
}

float minFixed(int k, int l)
{
	return (-1.0)*(float)powInt(2,k-1);
}

float floatToFixed(float number, int k, int l)
{
	float quant = 1.0/powInt(2,l);
	float div = roundInt(number/quant);
	float ret = quant*div;
	assert(ret <= maxFixed(k,l) && ret >= minFixed(k,l));
//	if (ret > maxFixedPoint(k,l) || ret < minFixedPoint(k,l))
//	{
//		printf("\nWARNING! Can not convert %f to a <%i,%i> bits fixed point. Insufficient range!", number, k, l);
//		if (number > 0) printf("\nUse %i bits for integer part.", (int)ceil(log2(number)+1.0));
//		else printf("\nUse %i bits for integer part.", (int)ceil(log2(-number)+1.0));
//		exit(1);
//	}
	return ret;
}

float iirOut(float y[], float x[], float a[], float b[], int Na, int Nb)
{																			// timer1 += 8;
	float *a_ptr, *y_ptr, *b_ptr, *x_ptr;									// timer1 += 4;
	float a_sum = 0, b_sum = 0;												// timer1 += 5;
	a_ptr = &a[1];															// timer1 += 1;
	y_ptr = &y[Na-1];														// timer1 += 5;
	b_ptr = &b[0];															// timer1 += 1;
	x_ptr = &x[Nb-1];														// timer1 += 6;
	int k, j;
	for (k = 0; k < Nb; k++){												// timer1 += 3;
		b_sum += *b_ptr++ * *x_ptr--;										// timer1 += (5+3+3+3);
	}
	for (j = 0; j < Na-1; j++){												// timer1 += 2;
		a_sum -= *a_ptr++ * *y_ptr--;										// timer1 += (5+5+3+1);
	}																		// timer1 += 4;
	return a_sum + b_sum;													// timer1 += 4;
}																			// timer1 += 3;

int timer1 = 0;
float x_fixed, y_fixed, sum_fixed;
float iirOutTimer(float y[], float x[], float a[], float b[], int Na, int Nb)
{																			// timer1 += 8;
	timer1 = 0;
	float *a_ptr, *y_ptr, *b_ptr, *x_ptr;									// timer1 += 4;
	float a_sum = 0, b_sum = 0;												// timer1 += 5;
	a_ptr = &a[1];															// timer1 += 1;
	y_ptr = &y[Na-1];														// timer1 += 5;
	b_ptr = &b[0];															// timer1 += 1;
	x_ptr = &x[Nb-1];														// timer1 += 6;
	int k, j;
	timer1 += 30;//(8+4+5+1+5+1+6);
	for (k = 0; k < Nb; k++){												// timer1 += 3;
		//x_fixed = floatToFixed(*x_ptr--, 2, 5);
		b_sum += *b_ptr++ * *x_ptr--;										// timer1 += (5+3+3+3);
		//b_sum = floatToFixed(b_sum, 2, 5);
		timer1 += 17;//(5+3+3+3+3);
	}
	for (j = 0; j < Na-1; j++){
		//y_fixed = floatToFixed(*y_ptr--, 2, 5); 							// timer1 += 2;
		a_sum -= *a_ptr++ * *y_ptr--;										// timer1 += (5+5+3+1);
		//a_sum = floatToFixed(a_sum, 2, 5);
		timer1 += 16;//(5+5+3+1+2);
	}																		// timer1 += 4;
	timer1 += 11;//(4+4+3);
	assert(timer1<100000);
	float ret = a_sum + b_sum;
	sum_fixed = floatToFixed(ret, 2, 5);
	return ret;																// timer1 += 4;
}																			// timer1 += 3;


float nondet_float();

int checkOverflow(void)
{
	int i;
	int xsize = 16; // Quantidade de entradas. Mesma quantidade de loops do filtro
	int Na = 3;		// Quantidade de coeficientes do denominador
	int Nb = 3;		// Quantidade de coeficientes do numerador
	int k = 2; 		// Numero de bits da parte inteira
	int l = 5; 		// Numero de bits da parte fracionaria
	float max = 1.7;//maxFixed(k,l);
	float min = -1.7;//minFixed(k,l);

	float y[xsize];
	float x[xsize];
//	float *y = (float *)malloc(xsize*sizeof(float));
//	memset(y, 0, sizeof(float) * xsize);
//	float *x = (float *)malloc(xsize*sizeof(float));
//	memset(x, 0, sizeof(float) * xsize);
	for (i = 0; i<xsize; ++i)
	{
		y[i] = 0;
		x[i] = nondet_float();
		__ESBMC_assume(x[i]>=min && x[i]<=max);
	}

	float yaux[Na];
	float xaux[Nb];
//	float *yaux = (float *)malloc(Na*sizeof(float));
//	memset(yaux, 0, sizeof(float) * Na);
//	float *xaux = (float *)malloc(Nb*sizeof(float));
//	memset(xaux, 0, sizeof(float) * Nb);
	for (i = 0; i<Na; ++i)
	{
		yaux[i] = 0;
	}
	for (i = 0; i<Nb; ++i)
	{
		xaux[i] = 0;
	}

	float a[Na];
	float b[Nb];
//	float *a = (float *)malloc(Na*sizeof(float));
//	memset(a, 0, sizeof(float) * Na);
//	float *b = (float *)malloc(Nb*sizeof(float));
//	memset(b, 0, sizeof(float) * Nb);

//	//Coefficients for Notch filter
//	a[0] = 1.0000; a[1] = -0.4665; a[2] = 0.5095;
//	b[0] = 0.7548; b[1] = -0.4665; b[2] = 0.7548;

	a[0] = 1.0000; a[1] = -0.375; a[2] = 0.1875;
	b[0] = 0.21875; b[1] = 0.40625; b[2] = 0.21875;

assert(0);
	////////////// Filter //////////////
	for(i=0; i<xsize; ++i)
	{
		shiftL(x[i],xaux,Nb);
		y[i] = iirOutTimer(yaux,xaux,a,b,Na,Nb);
		shiftL(y[i],yaux,Na);
	}

//	free(x);
//	free(y);
//	free(xaux);
//	free(yaux);
//	free(a);
//	free(b);
	return 0;
}

int checkLimitCycle(void)
{
	int i;
	int xsize = 16;
	int Na = 3;
	int Nb = 3;

	float y[xsize];
	float x[xsize];
//	float *y = (float *)malloc(xsize*sizeof(float));
//	memset(y, 0, sizeof(float) * xsize);
//	float *x = (float *)malloc(xsize*sizeof(float));
//	memset(x, 0, sizeof(float) * xsize);
	for (i = 0; i<xsize; ++i)
	{
		y[i] = 0;
		x[i] = 0;
	}

	float yaux[Na];
	float xaux[Nb];
//	float *yaux = (float *)malloc(Na*sizeof(float));
//	memset(yaux, 0, sizeof(float) * Na);
//	float *xaux = (float *)malloc(Nb*sizeof(float));
//	memset(xaux, 0, sizeof(float) * Nb);
	for (i = 0; i<Na; ++i)
	{
		yaux[i] = nondet_float();
		__ESBMC_assume(yaux[i]>-1.0 && yaux[i]<1.0);
	}
	for (i = 0; i<Nb; ++i)
	{
		xaux[i] = 0;
		///// VERIFICAR SE PRECISA INICIAR xaux[i] = nondet_float();
		//xaux[i] = nondet_float();
	}

	float a[Na];
	float b[Nb];
//	float *a = (float *)malloc(Na*sizeof(float));
//	memset(a, 0, sizeof(float) * Na);
//	float *b = (float *)malloc(Nb*sizeof(float));
//	memset(b, 0, sizeof(float) * Nb);
	a[0] = 1.0000; a[1] = -0.4665; a[2] = 0.5095;
	b[0] = 0.7548; b[1] = -0.4665; b[2] = 0.7548;

	int window = order(Na,Nb);
	int j;
	int repeated = 0;

	////////////// Filter //////////////
	for(i=0; i<xsize; ++i)
	{
		//// VERIFICANDO REPETICAO DA JANELA DO TAMANHO DA ORDEM DO FILTRO
		if(i%window == 0 && i>=2*window)
		{
			for(j=i-window; j<i; ++j)
			{
				repeated += (y[j]==y[j-window] && y[j] != 0);
			}
			assert(repeated < window);
			repeated = 0;
		}

		shiftL(x[i],xaux,Nb);
		y[i] = iirOut(yaux,xaux,a,b,Na,Nb);
		shiftL(y[i],yaux,Na);
	}

//	free(x);
//	free(y);
//	free(xaux);
//	free(yaux);
//	free(a);
//	free(b);
	return 0;
}
