int main()
{
  float a[2];
  int temp;
  a[0] = -0.375f;
	temp = (int)(a[0]/0.03125f);
	assert(temp==-12);
}

