%{
  // A parser for smtlib responses

#include "smtlib_conv.h"

#include "y.tab.hpp"

int smtliblex();
int smtliberror(const std::string &error);

%}

/* Values */
%union {
  const char *text;
};

/* Some tokens */

/* Start token, for the response */
%start response

/* Types */

%%

/* Rules */

response:
