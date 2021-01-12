#define a 2
int main()
{
  unsigned long long int i=1, sn=0;
  while ( 1 ) {
    sn = sn + ((i%10==9)? 4 : a);
    i++;
    assert ( sn==(i-1)*a ) ;
  }
}
