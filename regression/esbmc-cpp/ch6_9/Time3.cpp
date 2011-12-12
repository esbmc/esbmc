// Fig. 6.19: time3.cpp
// Member-function definitions for Time class.
#include <iostream>

using std::cout;

#include <iomanip>

using std::setfill;
using std::setw;

// include definition of class Time from time3.h
#include "time3.h"

// constructor function to initialize private data;
// calls member function setTime to set variables;
// default values are 0 (see class definition)
Time::Time( int hr, int min, int sec ) 
{ 
   setTime( hr, min, sec ); 

} // end Time constructor

// set hour, minute and second values
void Time::setTime( int h, int m, int s )
{
   setHour( h );
   setMinute( m );
   setSecond( s );

} // end function setTime

// set hour value
void Time::setHour( int h ) 
{
   hour = ( h >= 0 && h < 24 ) ? h : 0; 

} // end function setHour

// set minute value
void Time::setMinute( int m )
{ 
   minute = ( m >= 0 && m < 60 ) ? m : 0; 

} // end function setMinute

// set second value
void Time::setSecond( int s )
{ 
   second = ( s >= 0 && s < 60 ) ? s : 0; 

} // end function setSecond

// return hour value
int Time::getHour() 
{ 
   return hour; 

} // end function getHour

// return minute value
int Time::getMinute() 
{
   return minute; 

} // end function getMinute

// return second value
int Time::getSecond() 
{ 
   return second;
   
} // end function getSecond

// print Time in universal format
void Time::printUniversal()
{
   cout << setfill( '0' ) << setw( 2 ) << hour << ":"
        << setw( 2 ) << minute << ":"
        << setw( 2 ) << second;

} // end function printUniversal

// print Time in standard format
void Time::printStandard()
{
   cout << ( ( hour == 0 || hour == 12 ) ? 12 : hour % 12 )
        << ":" << setfill( '0' ) << setw( 2 ) << minute
        << ":" << setw( 2 ) << second 
        << ( hour < 12 ? " AM" : " PM" );

} // end function printStandard

/**************************************************************************
 * (C) Copyright 1992-2002 by Deitel & Associates, Inc. and Prentice      *
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
