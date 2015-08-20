int main()
{
  float result, a[3];
  a[0] = 1.0000; a[1] = -0.375; a[2] = 0.1875;
  result = a[0]+a[1]+a[2];
  assert(result==0.8126);

}
