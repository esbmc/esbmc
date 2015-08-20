/*
 * EmbeddedPC.cpp
 *
 *  Created on: Jan 17, 2012
 *      Author: mikhail
 */

#include "EmbeddedPC.h"

#include <iostream>
#include <cmath>
#include <sstream>

#define VIAGEM_MSG "Distance traveled: "
#define VELOCIDADE_MSG "Current speed: "
#define TOTAL_MSG "Total distance so far: "
#define TEMPO_MSG "Time elapsed since started: "

EmbeddedPC::EmbeddedPC() :
	_m(VIAGEM),
	_battery(false),
	_dist(0),
	_totalDist(0),
	_speed(0)
{
	gettimeofday(&_begin, NULL);
	_lastWheelTurn = _begin;

	pthread_mutex_init(&mutex, NULL);

	std::stringstream msg;
	//msg << std::string(VIAGEM_MSG);
	//msg << " " << _dist << " meters";

	_d = new Display(msg.str());
	_d->lock();
	_d->setRefreshRate(200); // begin on Viagem mode
	_d->start();
	_d->unlock();
}

void EmbeddedPC::resetPressed()
{
	switch(_m)
	{
		case VIAGEM:
			_dist = 0;
			break;

		case TEMPO:
			gettimeofday(&_begin, NULL);
			break;

		case TOTAL:
		case VELOCIDADE:
			break;
	}
}

void EmbeddedPC::modePressed()
{
	_m = Mode((_m + 1) % 4);
	updateDisplay();
}

std::string EmbeddedPC::currentMode()
{
	std::stringstream msg;
//	msg << "(MODE: ";

	switch(_m)
	{
		case VIAGEM:
			msg << "VIAGEM)";
			break;

		case VELOCIDADE:
			msg << "VELOCIDADE)";
			break;

		case TOTAL:
			msg << "TOTAL)";
			break;

		case TEMPO:
			msg << "TEMPO)";
			break;
	}

	return msg.str();
}

void EmbeddedPC::updateDisplay()
{
	_d->lock();

	std::stringstream msg;
	switch(_m)
	{
		case VIAGEM:
			msg << VIAGEM_MSG;
			msg << " " << _dist << " meters";

			_d->setRefreshRate(200);
			break;

		case VELOCIDADE:
			msg << VELOCIDADE_MSG;
			msg << " " << _speed << " m/s";

			_d->setRefreshRate(100);
			break;

		case TOTAL:
			msg << TOTAL_MSG;
			msg << " " << _totalDist << " meters";

			_d->setRefreshRate(500);
			break;

		case TEMPO:
			struct timeval now;
			gettimeofday(&now, NULL);
			float time = now.tv_sec - _begin.tv_sec;

			msg << TEMPO_MSG;
			msg << " " << time << " s";

			_d->setRefreshRate(500);
			break;
	}
	_d->setMessage(msg.str());

	_d->unlock();
}

void EmbeddedPC::wheelTurn()
{
	struct timeval now;
	gettimeofday(&now, NULL);

	float time = std::abs(now.tv_usec - _lastWheelTurn.tv_usec)/1000000;
	__ESBMC_assume(time!=0);
	// each wheel turn, 1m is moved
	_dist += 1;
	_totalDist += 1;
	_speed = 1/time;
	gettimeofday(&_lastWheelTurn, NULL);

	updateDisplay();
}
