#include <string>
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
  QApplication a(argc, argv);
  QGraphicsScene s;
  QGraphicsView v(&s);

  Background background("Butterfly.bmp");
  s.addItem(&background);
  v.show();
  return a.exec();
}
