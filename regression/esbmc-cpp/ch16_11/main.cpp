// Fig. 16.24: logout.cpp
// Program to log out of the system.
#include <iostream>

using std::cout;

#include <string>

using std::string;

#include <ctime>

#include <cstdlib>

void writeCookie();

int main()
{
   // write the cookie
   writeCookie();

   // output header
   cout << "Content-type: text/html\n\n";

   // output XML declaration and DOCTYPE
   cout << "<?xml version = \"1.0\"?>" 
        << "<!DOCTYPE html PUBLIC \"-//W3C//DTD XHTML 1.0 "
        << "Transitional//EN\"";
   
   // output html element and its contents
   cout << "<html xmlns = \"http://www.w3.org/1999/xhtml\">"
        << "<head><title>Logged Out</title></head>"
        << "<body>"
        << "<center><p>You are now logged out<br>"
        << "You will be billed accordingly<br>"
        << "To login again, "
        << "<a href=\"/cgi-bin/login.cgi\">click here</a>"
        << "</body></html>\n";

   return 0;

} // end main

// function to write cookie
void writeCookie()
{
   // string containing expiration date
   string expires = "Friday, 14-MAY-10 16:00:00 GMT";

   // set cookie
   cout << "set-cookie: CART=; expires=" << expires 
        << "; path=\n";
  
} // end writeCookie

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