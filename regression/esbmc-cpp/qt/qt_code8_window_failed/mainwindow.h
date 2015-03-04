#ifndef MAINWINDOW_H
#define MAINWINDOW_H
//#include <QDialog>
#include <QtGui>
class QPushButton;
class QLabel;
class QSpinBox;
class QSlider;
class QGridLayout;
class QVBoxLayout;
class QHBoxLayout;
class MainWindow : public QWidget{
	Q_OBJECT
	public :
	MainWindow ( QWidget* parent = 0 );
	private slots :
	void resetSliders ( );
	private :
	QPushButton *botaoReset;
	QPushButton *botaoSair;
	QLabel *textoRed;
	QLabel *textoGreen;
	QLabel *textoBlue;
	QSpinBox *spinBoxRed;
	QSpinBox *spinBoxGreen;
	QSpinBox *spinBoxBlue;
	QSlider *sliderRed;
	QSlider *sliderGreen;
	QSlider *sliderBlue;
};
#endif

