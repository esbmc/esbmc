// Fig. 7.30: scheduler.cpp
// Member-function definitions for class Scheduler.
#include <iostream>

using std::cout;
using std::endl;

#include <new>
#include <cstdlib>
#include <ctime>

#include "scheduler.h"  // Scheduler class definition
#include "floor.h"      // Floor class definition
#include "person.h"     // Person class definition

// constructor
Scheduler::Scheduler( Floor &firstFloor, Floor &secondFloor )
   : currentClockTime( 0 ),  
     floor1Ref( firstFloor ),
     floor2Ref( secondFloor )
{
   srand( time( 0 ) ); // seed random number generator
   cout << "scheduler constructed" << endl;

   // schedule first arrivals for floor 1 and floor 2
   scheduleTime( floor1Ref );
   scheduleTime( floor2Ref );

} // end Scheduler constructor

// destructor
Scheduler::~Scheduler() 
{ 
   cout << "scheduler destructed" << endl; 

} // end Scheduler destructor

// schedule arrival on a floor
void Scheduler::scheduleTime( const Floor &floor )
{
   int floorNumber = floor.getNumber();
   int arrivalTime = currentClockTime + ( 5 + rand() % 16 );

   floorNumber == Floor::FLOOR1 ?
      floor1ArrivalTime = arrivalTime : 
      floor2ArrivalTime = arrivalTime;

   cout << "(scheduler schedules next person for floor " 
        << floorNumber << " at time " << arrivalTime << ')'
        << endl;

} // end function scheduleTime

// reschedule arrival on a floor
void Scheduler::delayTime( const Floor &floor )
{
   int floorNumber = floor.getNumber();

   int arrivalTime = ( floorNumber == Floor::FLOOR1 ) ? 
      ++floor1ArrivalTime : ++floor2ArrivalTime;

   cout << "(scheduler delays next person for floor "
        << floorNumber << " until time " << arrivalTime << ')'
        << endl;

} // end function delayTime

// give time to scheduler
void Scheduler::processTime( int time )
{
   currentClockTime = time;   // record time
   
   // handle arrivals on floor 1
   handleArrivals( floor1Ref, currentClockTime );

   // handle arrivals on floor 2
   handleArrivals( floor2Ref, currentClockTime );
    
} // end function processTime

// create new person and place it on specified floor
void Scheduler::createNewPerson( Floor &floor )
{
   int destinationFloor = 
      floor.getNumber() == Floor::FLOOR1 ?
         Floor::FLOOR2 : Floor::FLOOR1;

   // create new person
   Person *newPersonPtr = new Person( destinationFloor );

   cout << "scheduler creates person " 
        << newPersonPtr->getID() << endl;
   
   // place person on proper floor
   newPersonPtr->stepOntoFloor( floor );
   
   scheduleTime( floor ); // schedule next arrival

} // end function createNewPerson

// handle arrivals for a specified floor
void Scheduler::handleArrivals( Floor &floor, int time )
{
   int floorNumber = floor.getNumber();

   int arrivalTime = ( floorNumber == Floor::FLOOR1 ) ? 
      floor1ArrivalTime : floor2ArrivalTime;

   if ( arrivalTime == time ) {
      
      if ( floor.isOccupied() )     // if floor occupied,
         delayTime( floor );        // delay arrival

      else                          // otherwise, 
         createNewPerson( floor );  // create new person

   } // end outer if

} // end function handleArrivals

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
