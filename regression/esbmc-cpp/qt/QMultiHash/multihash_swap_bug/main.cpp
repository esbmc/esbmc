#include <cassert>
#include <QMultiHash>

int main ()
{
    QMultiHash<int, int> hash1;
    QMultiHash<int, int> hash2;
    QMultiHash<int, int> :: const_iterator it;

    hash1.insert(1,500);
    hash1.insert(2,300);
    hash1.insert(3,100);

    hash2.insert(1,900);
    hash2.insert(2,800);
    hash2.insert(3,700);

    hash1.swap(hash2);

    it = hash1.find(2,300);

    assert(it.key() == 2);
    assert(it.value() == 800);

    return 0;
}

