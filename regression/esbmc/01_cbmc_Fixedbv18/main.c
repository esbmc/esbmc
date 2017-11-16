int main()
{
  float a[2];
  unsigned int temp;
  a[0] = 0.375f;
	temp = (unsigned int)(a[0]/0.03125f);
	assert(temp==12);
}
