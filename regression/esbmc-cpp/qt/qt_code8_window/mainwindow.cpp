//#include <QtGui>
#include "mainwindow.h"
MainWindow::MainWindow (QWidget *parent):QWidget(parent){
	//objects
	botaoReset	= new QPushButton("Reset");
	botaoSair	= new QPushButton("Sair");
	textoRed	= new QLabel("Red");
	textoGreen	= new QLabel("Green");
	textoBlue	= new QLabel("Blue");
	spinBoxRed	= new QSpinBox;
	spinBoxGreen= new QSpinBox;
	spinBoxBlue	= new QSpinBox;
	//properties
	spinBoxRed->setRange(0, 255);
	spinBoxGreen->setRange (0, 255);
	spinBoxBlue->setRange (0, 255);
	sliderRed = new QSlider(Qt::Horizontal);
	sliderGreen = new QSlider(Qt::Horizontal);
	sliderBlue = new QSlider(Qt::Horizontal);
	sliderRed->setRange(0, 255);
	sliderGreen->setRange(0, 255);
	sliderBlue->setRange(0, 255);
	// Signals and slots :
	connect ( spinBoxRed, SIGNAL(valueChanged( int)), sliderRed, SLOT(setValue (int)));
	connect ( sliderRed, SIGNAL(valueChanged( int)), spinBoxRed, SLOT(setValue (int)));
	connect ( spinBoxGreen, SIGNAL(valueChanged(int )), sliderGreen, SLOT(setValue (int)));
	connect ( sliderGreen, SIGNAL(valueChanged(int)), spinBoxGreen, SLOT(setValue (int)));
	connect ( spinBoxBlue, SIGNAL(valueChanged(int)), sliderBlue, SLOT(setValue (int)));
	connect ( sliderBlue, SIGNAL(valueChanged( int)), spinBoxBlue, SLOT(setValue (int )));
	connect ( botaoSair, SIGNAL(clicked()), this, SLOT(close ()));
	connect ( botaoReset, SIGNAL(clicked()), this, SLOT(resetSliders()));
	// Layouts :
	QGridLayout *layoutCores = new QGridLayout;
	layoutCores->addWidget(textoRed, 0, 0, 1, 1);
	layoutCores->addWidget(spinBoxRed, 0, 1, 1,1);
	layoutCores->addWidget(sliderRed,0, 2, 1, 1);
	layoutCores->addWidget(textoGreen, 1, 0, 1, 1);
	layoutCores->addWidget(spinBoxGreen, 1, 1, 1, 1);
	layoutCores->addWidget(sliderGreen, 1, 2, 1, 1);
	layoutCores->addWidget(textoBlue,2, 0, 1, 1 );
	layoutCores->addWidget(spinBoxBlue, 2, 1, 1, 1);
	layoutCores->addWidget(sliderBlue, 2, 2, 1, 1);

	QHBoxLayout *layoutControlo = new QHBoxLayout;
	layoutControlo->addWidget(botaoReset);
	layoutControlo->addWidget(botaoSair);

	QVBoxLayout *layoutJanela = new QVBoxLayout;
	layoutJanela->addLayout(layoutCores);
	layoutJanela->addLayout(layoutControlo);
	setLayout(layoutJanela);
}

void MainWindow::resetSliders (){
	sliderRed->setValue (0);
	sliderGreen->setValue (0);
	sliderBlue->setValue (0);
}
