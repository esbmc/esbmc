/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include<iostream>
#include "var_name.h"

/*******************************************************************\

Function: get_variable_name

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

char* get_variable_name(std::string str)
{
  int j=0;
  char result[str.length()];

  for(int i=str.length()-1; i>=0; i--)
  {
	if (str.compare(i,1,":") == 0)
	{
	  for(int k=0; k<j; k++)
		result[k] = str[i+k+1];
	  break;
	}
	j++;
  }

  result[j] = str[str.length()];

  return result;
}
