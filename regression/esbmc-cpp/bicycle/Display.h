/*
 * Display.h
 *
 *  Created on: Jan 17, 2012
 *      Author: mikhail
 */

#ifndef DISPLAY_H_
#define DISPLAY_H_

#include <fstream>
#include <string>

#include "Thread.h"

class Display: public Thread {
public:
	Display(std::string msg);
	void setRefreshRate(int refreshRate);
	void setMessage(std::string msg);

private:
	void run();

private:
	std::ofstream _display;
	std::string _msg;
	int _refreshRate;
};

#endif /* DISPLAY_H_ */
