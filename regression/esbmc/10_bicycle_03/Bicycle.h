/*
 * Bicycle.h
 *
 *  Created on: Jan 17, 2012
 *      Author: mikhail
 */

#ifndef BICYCLE_H_
#define BICYCLE_H_

#include "Thread.h"
#include "EmbeddedPC.h"

class Bicycle : public Thread
{
public:
	Bicycle(EmbeddedPC *pc);
	Bicycle(EmbeddedPC *pc, int vel);

	void turnOffPc();

private:
	void run();

private:
	EmbeddedPC *_pc;
};

#endif /* BICYCLE_H_ */
