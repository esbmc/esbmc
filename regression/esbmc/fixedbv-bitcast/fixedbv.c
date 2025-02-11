
int main()
{
	float f = 1;
	unsigned int *p = &f;
	unsigned int v = *p;
	assert(v == 1 << 16);
}
