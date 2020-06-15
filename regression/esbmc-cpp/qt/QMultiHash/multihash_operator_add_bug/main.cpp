##include <cassert>
#include <QMultiHash>

int main ()
{
    QMultiHash<int, int> hash1;
    QMultiHash<int, int> hash2;
    QMultiHash<int, int> hash3;
    QMultiHash<int, int> :: const_iterator it;

    hash1.insert(1,500);
    hash1.insert(2,300);
    hash1.insert(3,100);

    hash2.insert(4,900);
    hash2.insert(5,800);
    hash2.insert(6,700);

    hash3 = hash1 + hash2;

    assert(hash3.size() < 6);

    return 0;
}
