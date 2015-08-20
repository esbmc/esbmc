//#include <QApplication>
#include "mainwindow.h"
int main ( int argc, char *argv [] ){
	QApplication app ( argc, argv );
	MainWindow *janela = new MainWindow;
	janela->setWindowTitle ("Modelo RGB!");
	janela->show();
	return app.exec();
}
