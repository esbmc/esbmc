#include <QtGui>

int main(int argv, char **args)
{

    QApplication app(argv, args);

    QTextEdit textEdit;
    QPushButton quitButton("Quit");

    QObject::connect(&quitButton, SIGNAL(clicked()), qApp, SLOT(quit()));

    QVBoxLayout layout;
    layout.addWidget(&textEdit);
    layout.addWidget(&quitButton);

    QWidget window;
    window.setLayout(&layout);

    window.show();

    return app.exec();

}
