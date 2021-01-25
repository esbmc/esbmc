#define TRUE 1
#define FALSE 0

int roundInt(float number)
{
	float ret = (number >= 0) ? (number + 0.5) : (number - 0.5);
	assert((int)ret != -13.0f); //DEVERIA SER 12.0
    return (int)ret;
}

float floatToFix(float number, float delta, int k, int l, int catchOverflow)
{
	float div = roundInt(number/delta);
	float ret = delta*div;
	return ret;
}

int main(void)
{
	int i;
	int Na = 3;
	int Nb = 3;
	int k = 2;
	int l = 5;
	float delta = 0.03125;

	float a[Na];
	float b[Nb];

	a[0] = 1.0000; a[1] = -0.375; a[2] = 0.1875;
	b[0] = 0.21875; b[1] = 0.40625; b[2] = 0.21875; // SE COMENTAR ESSA LINHA OU COLOCA-LA DEPOIS DO for O ASSERT DA LINHA 7 PASSA

	// QUANTIZANDO COEFICIENTES. NESSE CASO DEVE DAR O MESMO VALOR.
	for (i=0; i<Na; ++i)
	{
		a[i] = floatToFix(a[i], delta, k, l, FALSE);
	}

	return 0;
}

