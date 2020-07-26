#include <QMap>
#include <cassert>

int main ()
{
  QMap<int,int> myQMap;
  QMap<int,int>::iterator it;

  myQMap[1] = 100;
  myQMap[2] = 200;
  myQMap[3] = 300;

  it = myQMap.begin();
  myQMap.equal_range(it.key());

  assert(it.key() != 1);
  assert(it.value() != 100);
  it++;
  assert(it.key() != 2);
  assert(it.value() != 200);
  it++;
  assert(it.key() != 3);
  assert(it.value() != 300);


  return 0;
}

