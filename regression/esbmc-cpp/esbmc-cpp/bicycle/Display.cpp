/*
 * Display.cpp
 *
 *  Created on: Jan 17, 2012
 *      Author: mikhail
 */

#include "Display.h"

Display::Display(std::string msg)
	: Thread(),
	  _msg(msg),
	  _refreshRate(0)
{
}

void Display::setRefreshRate(int r)
{
	if(r == 200 || r == 500 || r == 100 || r == 1000)
	{
		_refreshRate = r;
	}
}

void Display::setMessage(std::string msg)
{
	_msg = msg;
}

void Display::run()
{
	while(1)
	{
		msleep(_refreshRate);

		_display.open("Display", std::ios::out | std::ios::app );
		_display << _msg << ".\n";
		_display.close();
	}
}
