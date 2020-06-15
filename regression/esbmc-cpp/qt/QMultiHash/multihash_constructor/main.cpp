#include <cassert>
#include <QMultiHash>

int main ()
{
    QMultiHash<int, int> hash;
    assert(hash.size() == 0);

    return 0;
}
