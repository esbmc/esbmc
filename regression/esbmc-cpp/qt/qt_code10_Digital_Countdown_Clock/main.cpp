//#include <QApplication>
//#include <QMainWindow>
#include "LCDNumber.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QMainWindow *window = new QMainWindow();
    
    
        window->setWindowTitle(QString::fromUtf8("CountDown Clock"));
        window->resize(250, 250);
	
        QWidget *centralWidget = new QWidget(window);
	//Start Counting down from 5 minutes
	LCDNumber *number = new LCDNumber(centralWidget,5,0);	
	number->setFixedSize(245, 245);

	window->setCentralWidget(centralWidget);
	window->show();
	
	number->timer->start(1000);
	return app.exec();

}
