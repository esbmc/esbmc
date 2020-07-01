//main.cpp File

#include "LCDNumber.h"

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QMainWindow *window = new QMainWindow();    
    
        window->setWindowTitle(QString::fromUtf8("Qt QLCDnumber High Resolution Timer"));
        window->resize(800, 250);
	QWidget* centralWidget = new QWidget();
	
	QVBoxLayout* verticalLayout = new QVBoxLayout();
	centralWidget->setLayout(verticalLayout);
        
	//Start Counting down from 5 minutes
	LCDNumber *number = new LCDNumber(verticalLayout,window,5,0);	
	
	
	number->setFixedSize(-10, 245);
	number->setDigitCount(9);

	window->setCentralWidget(centralWidget);
	window->show();
	
	//QTimer resolution here is set to 1 millisecond
	//Depending on your system and system load this tick interval
	//may be in fact very long and your timer clock may not work 
	//accurately.
	//If you encounter such a problem try increasing QTimer resolution
	//and substract corresponding millisecond value from countdown
	number->timer->setInterval(1);
	number->timer->start();
	return app.exec();

}
