/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_MODE_H
#define CPROVER_MODE_H

#include <language.h>

struct mode_table_et
{
  const char *name;
  languaget *(*new_language)();
  const char **extensions;
};
 
extern const mode_table_et mode_table[];

int get_mode(const std::string &str);
int get_mode_filename(const std::string &filename);

languaget *new_language(int mode);

#define MODE_C        0
#define MODE_IREP     1
#define MODE_PVS      2
#define MODE_VHDL     3
#define MODE_VERILOG  4
#define MODE_SMV      5
#define MODE_CONF     6
#define MODE_NETLIST  7
#define MODE_SPECC    8
#define MODE_PROMELA  9
#define MODE_XML      10
#define MODE_PASCAL   11
#define MODE_CPP      12
#define MODE_SIMPLIFY 13
#define MODE_BP       14
#define MODE_CVC      15
#define MODE_CSHARP   16
#define MODE_SMT      17
#define MODE_NSF      18
#define MODE_PHP      19
#define MODE_MDL      20

#endif
