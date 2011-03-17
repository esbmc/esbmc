/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <string.h>

// Prototypes

#include "mode.h"

const char *extensions_ansi_c  []={ "c", "i", NULL };
const char *extensions_intrep  []={ NULL };
const char *extensions_pvs     []={ "pvs", NULL };
const char *extensions_vhdl    []={ "vhdl", NULL };
const char *extensions_verilog []={ "v", NULL };
const char *extensions_smv     []={ "smv", "flat_smv", NULL };
const char *extensions_csp     []={ "csp", NULL };
const char *extensions_netlist []={ "ntl", NULL };
const char *extensions_conf    []={ "conf", NULL };
const char *extensions_specc   []={ "sc", "si", NULL };
const char *extensions_promela []={ "promela", NULL };
const char *extensions_xml     []={ "xmi", "xml", NULL };
const char *extensions_pascal  []={ "pas", NULL };

#ifdef _WIN32
const char *extensions_cpp     []={ "cpp", "cc", "ipp", "cxx", NULL };
#else
const char *extensions_cpp     []={ "cpp", "cc", "ipp", "C", "cxx", NULL };
#endif

const char *extensions_simplify[]={ "ax", NULL };
const char *extensions_bp      []={ "bp", NULL };
const char *extensions_cvc     []={ "cvc", NULL };
const char *extensions_csharp  []={ "cs", NULL };
const char *extensions_smt     []={ "smt", NULL };
const char *extensions_nsf     []={ "nsf", NULL };
const char *extensions_php     []={ "php", NULL };
const char *extensions_mdl     []={ "mdl", NULL };

languaget *new_ansi_c_language();
languaget *new_bp_language();
languaget *new_cpp_language();
languaget *new_csp_language();
languaget *new_csharp_language();
languaget *new_cvc_language();
languaget *new_intrep_language();
languaget *new_netlist_language();
languaget *new_pascal_language();
languaget *new_promela_language();
languaget *new_pvs_language();
languaget *new_simplify_language();
languaget *new_smv_language();
languaget *new_specc_language();
languaget *new_verilog_language();
languaget *new_vhdl_language();
languaget *new_smt_language();
languaget *new_nsf_language();
languaget *new_php_language();
languaget *new_mdl_language();

const mode_table_et mode_table[]=
{
  { "C",        &new_ansi_c_language,   extensions_ansi_c   }, // 0
  { "intrep",   &new_intrep_language,   extensions_intrep   }, // 1
  { "PVS",      &new_pvs_language,      extensions_pvs      }, // 2
  { "VHDL",     &new_vhdl_language,     extensions_vhdl     }, // 3
  { "Verilog",  &new_verilog_language,  extensions_verilog  }, // 4
  { "SMV",      &new_smv_language,      extensions_smv      }, // 5
  { "CSP",      &new_csp_language,      extensions_csp      }, // 6
  { "Netlist",  &new_netlist_language,  extensions_netlist  }, // 7
  { "SpecC",    &new_specc_language,    extensions_specc    }, // 8
  { "Promela",  &new_promela_language,  extensions_promela  }, // 9
  { "PASCAL",   &new_pascal_language,   extensions_pascal   }, // 11
  { "C++",      &new_cpp_language,      extensions_cpp      }, // 12
  { "Simplify", &new_simplify_language, extensions_simplify }, // 13
  { "bp",       &new_bp_language,       extensions_bp       }, // 14
  { "CVC",      &new_cvc_language,      extensions_cvc      }, // 15
  { "C#",       &new_csharp_language,   extensions_csharp   }, // 16
  { "SMT",      &new_smt_language,      extensions_smt      }, // 17
  { "NSF",      &new_nsf_language,      extensions_nsf      }, // 18
  { "PHP",      &new_php_language,      extensions_php      }, // 19
  { "MDL",      &new_mdl_language,      extensions_mdl      }, // 20
  { NULL,      NULL,                    NULL }
};

/*******************************************************************\

Function: get_mode

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_mode(const std::string &str)
{
  unsigned i;

  for(i=0; mode_table[i].name!=NULL; i++)
    #ifdef _WIN32
    if(strcasecmp(str.c_str(), mode_table[i].name)==0)
      return i;
    #else
    if(str==mode_table[i].name)
      return i;
    #endif

  return -1;
}

/*******************************************************************\

Function: get_mode_filename

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

int get_mode_filename(const std::string &filename)
{
  const char *ext=strrchr(filename.c_str(), '.');

  if(ext==NULL) return -1;

  std::string extension=ext+1;

  if(extension=="") return -1;

  int mode;
  for(mode=0; mode_table[mode].name!=NULL; mode++)
    for(unsigned i=0;
        mode_table[mode].extensions[i]!=NULL;
        i++)
      if(mode_table[mode].extensions[i]==extension)
        return mode;

  return -1;
}

/*******************************************************************\

Function: new_language

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

languaget *new_language(int mode)
{
  return (*mode_table[mode].new_language)();
}

