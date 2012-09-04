// Fig. 7.46: person.cpp
// Member-function definitions for class Person.
#include <iostream>

using std::cout;
using std::endl;

#include "person.h"    // Person class definition
#include "floor.h"     // Floor class definition
#include "elevator.h"  // Elevator class definition

// initialize static member personCount
int Person::personCount = 0;

// constructor
Person::Person( int destFloor )
   : ID( ++personCount ), 
     destinationFloor( destFloor ) 
{
   cout << "person " << ID << " constructed" << endl;

} // end Person constructor

// destructor
Person::~Person() 
{ 
   cout << "(person " << ID << " destructor invoked)" << endl;

} // end ~Person destructor

// return person's ID number
int Person::getID() const 
{ 
   return ID; 

} // end function getID

// person walks onto a floor
void Person::stepOntoFloor( Floor& floor )  
{
   // notify floor person is coming
   cout << "person " << ID << " steps onto floor " 
        << floor.getNumber() << endl;
   floor.personArrives( this );

   // press button on floor
   cout << "person " << ID 
        << " presses floor button on floor " 
        << floor.getNumber() << endl;
   floor.floorButton.pressButton();

} // end function stepOntoFloor

// person enters elevator
void Person::enterElevator( Elevator &elevator, Floor &floor )
{   
   floor.personBoardingElevator();   // person leaves floor

   elevator.passengerEnters( this ); // person enters elevator

   // press button on elevator
   cout << "person " << ID 
        << " presses elevator button" << endl;
   elevator.elevatorButton.pressButton();

} // end function enterElevator

// person exits elevator
void Person::exitElevator( 
   const Floor &floor, Elevator &elevator ) const
{
   cout << "person " << ID << " exits elevator on floor "
        << floor.getNumber() << endl;
   elevator.passengerExits();

} // end function exitElevator

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
