// Fig. 14.11: ClientData.cpp
// Class ClientData stores customer's credit information.
#include <iostream>
#include <string>
using std::string;

#include <cstring>
#include "clientData.h"

// default ClientData constructor
ClientData::ClientData( int accountNumberValue, 
   string lastNameValue, string firstNameValue,
   double balanceValue )
{
   setAccountNumber( accountNumberValue );
   setLastName( lastNameValue );
   setFirstName( firstNameValue );
   setBalance( balanceValue );

} // end ClientData constructor

// get account-number value
int ClientData::getAccountNumber() const
{
   return accountNumber;

} // end function getAccountNumber

// set account-number value
void ClientData::setAccountNumber( int accountNumberValue )
{
   accountNumber = accountNumberValue;

} // end function setAccountNumber

// get last-name value
string ClientData::getLastName() const
{
   return lastName;

} // end function getLastName

// set last-name value
void ClientData::setLastName( string lastNameString )
{
   // copy at most 15 characters from string to lastName
   const char *lastNameValue = lastNameString.data();
   int length = strlen( lastNameValue );
   length = ( length < 15 ? length : 14 );
   strncpy( lastName, lastNameValue, length );

   // append null character to lastName
   lastName[ length ] = '\0';

} // end function setLastName

// get first-name value
string ClientData::getFirstName() const
{
   return firstName;

} // end function getFirstName

// set first-name value
void ClientData::setFirstName( string firstNameString )
{
   // copy at most 10 characters from string to firstName
   const char *firstNameValue = firstNameString.data();
   int length = strlen( firstNameValue );
   length = ( length < 10 ? length : 9 );
   strncpy( firstName, firstNameValue, length );

   // append new-line character to firstName
   firstName[ length ] = '\0';

} // end function setFirstName

// get balance value
double ClientData::getBalance() const
{
   return balance;

} // end function getBalance

// set balance value
void ClientData::setBalance( double balanceValue )
{
   balance = balanceValue;

} // end function setBalance

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
