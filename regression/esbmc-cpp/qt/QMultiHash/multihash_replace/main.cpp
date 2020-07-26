#include <cassert>
#include <QMultiHash>

int main ()
{
    QMultiHash<int, int> hash1;
    QMultiHash<int, int> :: const_iterator it;

    hash1.insert(1,500);
    hash1.insert(2,300);
    hash1.insert(3,100);

    hash1.replace(2,900);

    it = hash1.find(2,900);

    assert(it.key() == 2);
    assert(it.value() == 900);

    return 0;
}
