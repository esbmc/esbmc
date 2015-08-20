#define a 2
int main()
{
  unsigned long long int i=1, sn=0;
  unsigned int n=70;
  while ( i<=n ) {
    sn = sn + ((i%55==54)? 4 : a);
    i++;
  }
  assert ( sn==n*a ) ;
}
