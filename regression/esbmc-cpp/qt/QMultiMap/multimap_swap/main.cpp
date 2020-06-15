// swap QMultiMaps
#include <iostream>
#include <QMultiMap>
#include <cassert>
using namespace std;

int main ()
{
  QMultiMap<char,int> foo;
  QMultiMap<char,int> bar;
  QMultiMap<char,int>::iterator it;

  foo['x']=100;
  foo['y']=200;

  bar['a']=11;
  bar['b']=22;
  bar['c']=33;

  foo.swap(bar);

  cout << "foo contains:\n";
  for ( it=foo.begin() ; it != foo.end(); it++ )
    cout << it.key() << " => " << it.value() << endl;

  cout << "bar contains:\n";
  for ( it=bar.begin() ; it != bar.end(); it++ )
    cout << it.key() << " => " << it.value() << endl;
    
  assert(bar['x']==100);
  assert(bar['y']==200);

  assert(foo['a']==11);
  assert(foo['b']==22);
  assert(foo['c']==33);

  return 0;
}
