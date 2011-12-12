int main()
{
  __int8 i1;
  __int16 i2;
  __int32 i3;
  __int64 i4;
  
  assert(sizeof(i1)==1);
  assert(sizeof(i2)==2);
  assert(sizeof(i3)==4);
  assert(sizeof(i4)==8);
}
