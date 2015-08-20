#define a 2
int main()
{
  unsigned long long int i=1, sn=0;
  while ( 1 ) {
    sn = sn + ((i%55==54)? 4 : a);
    assert ( sn==i*a ) ;
    i++;
  }
}
