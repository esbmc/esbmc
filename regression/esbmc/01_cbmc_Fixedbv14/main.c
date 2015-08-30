int roundInt(float number)
{
	float ret = (number >= 0.0f) ? (number + 0.5f) : (number - 0.5f);
	return (int)ret;
}

int main()
{
	float delta = 0.03125;
	float a[3];
	a[1] = -0.375;
	float b = roundInt(a[1]/delta);
	assert(b == -11.0f);
}
