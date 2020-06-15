#include <cassert>
#include <QHash>
#include <QString>
using namespace std;

int main ()
{
    QHash<QString, int> hash;
    assert(hash.size() == 0);
    return 0;
}