//main.cpp File

#include <QtGui>

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);
	QMainWindow *window = new QMainWindow();    
    
        window->setWindowTitle(QString::fromUtf8("QLCDnumber Show Milliseconds Time"));
        window->resize(800, 250);
	QWidget* centralWidget = new QWidget();
	
	QVBoxLayout* verticalLayout = new QVBoxLayout();
	centralWidget->setLayout(verticalLayout);
        
	QLCDNumber *number = new QLCDNumber();	
	
	verticalLayout->addWidget(number);
	
	
	number->setFixedSize(800, 245);
	number->setDigitCount(12);
	number->display(QTime::currentTime().toString(QString("hh:mm:ss.zzz")));
	
	
	window->setCentralWidget(centralWidget);
	window->show();
	return app.exec();
}
