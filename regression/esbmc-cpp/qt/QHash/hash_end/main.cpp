#include <cassert>
#include <QHash>
#include <QString>

int main ()
{
    QHash<QString, int> myQHash;
    QHash<QString, int> ::iterator it;

    myQHash["abc"] = 500;
    myQHash["cde"] = 300;
    myQHash["fgh"] = 100;

    it = myQHash.end();

    assert(myQHash.contains(it.key()) == true);

    return 0;
}
