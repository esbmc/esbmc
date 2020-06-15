#include <cassert>
#include <QMultiMap>

int main ()
{
    QMultiMap<int, int> multimap;
    assert(multimap.size() != 0);

    return 0;
}
