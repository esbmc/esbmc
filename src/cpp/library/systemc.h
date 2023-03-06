/***************************************************************
*		systemc.h
*	  
*		Includes the libraries to handle with SystemC
*		codes. Also defines the sc_main function
*     
*		@author Luciano Sobral <sobral.luciano@gmail.com>
*
***************************************************************/

#ifndef SYSTEMC_H
#define SYSTEMC_H

/***************************************************************/
/*********************** CPP INCLUDES **************************/
/***************************************************************/

//#include <iostream>
//#include <string>

/***************************************************************/
/*********************** SYSTEMC INCLUDES **********************/
/***************************************************************/

#include "systemc/kernel/sc_module.h"
#include "systemc/kernel/sc_wait.h"
//#include "systemc/kernel/sc_sensitive.h"

#include "systemc/communication/sc_signal.h"
#include "systemc/communication/sc_signal_ports.h"
#include "systemc/communication/sc_clock.h"

#define sc_main main

#endif
