#include <QList>
#include <cassert>
using namespace std;

int main ()
{
    QList<int> first;
    QList<int> second;
    first.push_back(10);
    second.append(first);
    assert(second.size() == 1);
    second.append(15);
    second.append(20);
    assert(second.size() == 3);
    return 0;
}
