#include <QtGui>
 
int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
	 char* str = NULL;
    QLabel label(str);
    label.show();
    return app.exec();
}
