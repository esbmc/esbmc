#include <cassert>
#include <QMultiMap>

int main ()
{
    QMultiMap<int, int> multimap1;
    QMultiMap<int, int> :: const_iterator it;

    multimap1.insert(1,500);
    multimap1.insert(2,300);
    multimap1.insert(3,100);
    
    QMultiMap<int, int> multimap2(multimap1);
    it = multimap2.constFind(1,500);

    assert(it.key() == 1);
    assert(it.value() == 500);

    return 0;
}
