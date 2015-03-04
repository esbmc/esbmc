// Fig. 7.44: floor.cpp
// Member-function definitions for class Floor.
#include <iostream>

using std::cout;
using std::endl;

#include "floor.h"     // Floor class definition
#include "person.h"    // Person class definition
#include "elevator.h"  // Elevator class definition
#include "door.h"      // Door class definition

// static constants that represent the floor numbers
const int Floor::FLOOR1 = 1;
const int Floor::FLOOR2 = 2;

// constructor
Floor::Floor(int number, Elevator &elevatorHandle ) 
   : floorButton( number, elevatorHandle ), 
     floorNumber( number ), 
     elevatorRef( elevatorHandle ),
     occupantPtr ( 0 ),
     light( floorNumber )     
{ 
   cout << "floor " << floorNumber << " constructed" << endl;

} // end Floor constructor

// destructor
Floor::~Floor() 
{ 
   delete occupantPtr;
   cout << "floor " << floorNumber << " destructed" << endl;

} // end ~Floor destructor

// determine whether floor is occupied
bool Floor::isOccupied() const 
{
   return ( occupantPtr != 0 ); 

} // end function isOccupied

// return this floor's number
int Floor::getNumber() const 
{ 
   return floorNumber; 

} // end function getNumber

// person arrives on floor
void Floor::personArrives( Person * const personPtr )
{ 
   occupantPtr = personPtr; 

} // end function personArrives

// notify floor that elevator has arrived
Person *Floor::elevatorArrived()
{
   cout << "floor " << floorNumber 
        << " resets its button" << endl;

   floorButton.resetButton();
   light.turnOn();

   return occupantPtr;  

} // end function elevatorArrived

// tell floor that elevator is leaving
void Floor::elevatorLeaving() 
{ 
   light.turnOff(); 

} // end function elevatorLeaving

// notifies floor that person is leaving
void Floor::personBoardingElevator() 
{ 
   occupantPtr = 0;  // person no longer on floor

} // end function personBoardingElevator

/**************************************************************************
 * (C) Copyright 1992-2003 by Deitel & Associates, Inc. and Prentice      *
 * Hall. All Rights Reserved.                                             *
 *                                                                        *
 * DISCLAIMER: The authors and publisher of this book have used their     *
 * best efforts in preparing the book. These efforts include the          *
 * development, research, and testing of the theories and programs        *
 * to determine their effectiveness. The authors and publisher make       *
 * no warranty of any kind, expressed or implied, with regard to these    *
 * programs or to the documentation contained in these books. The authors *
 * and publisher shall not be liable in any event for incidental or       *
 * consequential damages in connection with, or arising out of, the       *
 * furnishing, performance, or use of these programs.                     *
 *************************************************************************/
