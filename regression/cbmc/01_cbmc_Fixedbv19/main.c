int main()
{
  float a[2], temp;
  a[0] = -0.375f;
	temp = (int)(a[0]/0.03125f);
	assert(temp==-12.0f);
}

