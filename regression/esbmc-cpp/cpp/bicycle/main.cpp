//============================================================================
// Name        : lista5c.cpp
// Author      : Mikhail
// Version     :
// Copyright   :
// Description :
//============================================================================

#include <iostream>
#include <cctype>
#include <cstdlib>

#include "EmbeddedPC.h"
#include "Bicycle.h"

int main()
{
	std::cout << "Booting ...";

	// cria pc
	EmbeddedPC *pc = new EmbeddedPC;

	// cria bicicleta
	Bicycle *b = new Bicycle(pc);
	b->start(); // thread start

	std::cout << " Done!" << std::endl;

	std::cout << "----------------------------------------------------" << std::endl;
	std::cout << "You are now biking" << std::endl;
	std::cout << "Choose your option: " << std::endl;
	std::cout << "1 - Press Mode Button" << std::endl;
	std::cout << "2 - Press Reset Button" << std::endl;
	std::cout << "3 - Remove PC's Batteries" << std::endl;
	std::cout << "----------------------------------------------------" << std::endl;

	char op;
	while(1) {
		std::cout << pc->currentMode();
		std::cout << " Enter number: ";
		std::cin >> op;

		if(!isdigit(op))
		{
			std::cout << "Press 1, 2 or 3." << std::endl;
			continue;
		}

		switch(atoi(&op))
		{
			case 1:
				pc->modePressed();
				break;

			case 2:
				pc->resetPressed();
				break;

			case 3:
				std::cout << "Bye!" << std::endl;
				std::exit(1);
				break;

			default:
				std::cout << "Press 1, 2 or 3." << std::endl;
				break;
		}
	}

	return 0;
}
