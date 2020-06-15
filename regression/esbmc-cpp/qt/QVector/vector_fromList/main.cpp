#include <QVector>
#include <cassert>
#include <QList>
#include <QString>

int main ()
{
    QList<QString>list;
    list << "Sven" << "Kim" << "Ola";

    QVector<QString> vect = QVector<QString>::fromList(list);
    // vect: ["Sven", "Kim", "Ola"]

    assert( !(vect.isEmpty()) );

  return 0;
}

