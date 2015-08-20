#include <string>
//#include <QtGui/QApplication>
#include <QApplication>
#include <QGraphicsPixmapItem>
#include <QGraphicsScene>
#include <QGraphicsView>

struct Background : public QGraphicsPixmapItem
{
  Background(const std::string& filename)
  {
    QPixmap m(filename.c_str());
    this->setPixmap(m);
  }
};

int main(int argc, char *argv[])
{
  argc = -1;
  QApplication a(argc, argv);
  QGraphicsScene s;
  QGraphicsView v(&s);

  Background background("Butterfly.bmp");
  s.addItem(&background);
  v.show();
  return a.exec();
}
