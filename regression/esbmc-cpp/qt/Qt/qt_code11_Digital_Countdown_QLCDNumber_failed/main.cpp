//#include <QApplication>
 
#include "myWindow.h"

int main(int argc, char **argv)
{	
	QApplication app(argc, argv);

	MyWindow *window = new MyWindow();

		
        window->setWindowTitle(QString::fromUtf8("Change QLCDNumber Color"));
        window->resize(300,300);
	
	MyWindow win;
	win.decorate(window);

	return app.exec();
}
