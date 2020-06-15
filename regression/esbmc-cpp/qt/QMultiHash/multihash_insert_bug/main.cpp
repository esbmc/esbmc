#include <cassert>
#include <QMultiHash>

int main ()
{
    QMultiHash<int, int> hash1;
    QMultiHash<int, int> :: const_iterator it;

    hash1.insert(1,500);
    hash1.insert(2,300);
    hash1.insert(3,100);

    it = hash1.find(1,500);

    assert(it.key() != 1);
    assert(it.value() != 500);

    return 0;
}

