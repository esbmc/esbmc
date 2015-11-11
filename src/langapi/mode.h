/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_MODE_H
#define CPROVER_MODE_H

#include <language.h>

// Table recording details about langauge modes

struct mode_table_et
{
  const char *name;
  languaget *(*new_language)();
  const char **extensions;
};

// List of language modes that are going to be supported in the final tool.
// Must be declared by user of langapi, must end with HAVE_MODE_NULL.
 
extern const mode_table_et mode_table[];

extern const char *extensions_ansi_c[];
extern const char *extensions_intrep[];
extern const char *extensions_pvs[];
extern const char *extensions_vhdl[];
extern const char *extensions_smv[];
extern const char *extensions_csp[];
extern const char *extensions_netlist[];
extern const char *extensions_conf[];
extern const char *extensions_specc[];
extern const char *extensions_promela[];
extern const char *extensions_xml[];
extern const char *extensions_pascal[];
extern const char *extensions_cpp[];
extern const char *extensions_simplify[];
extern const char *extensions_bp[];
extern const char *extensions_cvc[];
extern const char *extensions_csharp[];
extern const char *extensions_smt[];
extern const char *extensions_nsf[];
extern const char *extensions_php[];
extern const char *extensions_mdl[];

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
languaget *new_vhdl_language();
languaget *new_smt_language();
languaget *new_nsf_language();
languaget *new_php_language();
languaget *new_mdl_language();

// List of language entries, one can put in the mode table:

#define LANGAPI_HAVE_MODE_C \
  { "C",        &new_ansi_c_language,   extensions_ansi_c   }
#define LANGAPI_HAVE_MODE_INTREP \
  { "intrep",   &new_intrep_language,   extensions_intrep   }
#define LANGAPI_HAVE_MODE_PVS \
  { "PVS",      &new_pvs_language,      extensions_pvs      }
#define LANGAPI_HAVE_MODE_VHDL \
  { "VHDL",     &new_vhdl_language,     extensions_vhdl     }
#define LANGAPI_HAVE_MODE_SMV \
  { "SMV",      &new_smv_language,      extensions_smv      }
#define LANGAPI_HAVE_MODE_CSP \
  { "CSP",      &new_csp_language,      extensions_csp      }
#define LANGAPI_HAVE_MODE_NETLIST \
  { "Netlist",  &new_netlist_language,  extensions_netlist  }
#define LANGAPI_HAVE_MODE_SPECC \
  { "SpecC",    &new_specc_language,    extensions_specc    }
#define LANGAPI_HAVE_MODE_PROMELA \
  { "Promela",  &new_promela_language,  extensions_promela  }
#define LANGAPI_HAVE_MODE_PASCAL \
  { "PASCAL",   &new_pascal_language,   extensions_pascal   }
#define LANGAPI_HAVE_MODE_CPP \
  { "C++",      &new_cpp_language,      extensions_cpp      }
#define LANGAPI_HAVE_MODE_SIMPLIFY \
  { "Simplify", &new_simplify_language, extensions_simplify }
#define LANGAPI_HAVE_MODE_BP \
  { "bp",       &new_bp_language,       extensions_bp       }
#define LANGAPI_HAVE_MODE_CVC \
  { "CVC",      &new_cvc_language,      extensions_cvc      }
#define LANGAPI_HAVE_MODE_CSHARP \
  { "C#",       &new_csharp_language,   extensions_csharp   }
#define LANGAPI_HAVE_MODE_SMT \
  { "SMT",      &new_smt_language,      extensions_smt      }
#define LANGAPI_HAVE_MODE_NSF \
  { "NSF",      &new_nsf_language,      extensions_nsf      }
#define LANGAPI_HAVE_MODE_PHP \
  { "PHP",      &new_php_language,      extensions_php      }
#define LANGAPI_HAVE_MODE_MDL \
  { "MDL",      &new_mdl_language,      extensions_mdl      }
#define LANGAPI_HAVE_MODE_END {NULL, NULL, NULL}

int get_mode(const std::string &str);
int get_mode_filename(const std::string &filename);

languaget *new_language(const char *mode);

#define MODE_C        "C"
#define MODE_IREP     "intrep"
#define MODE_PVS      "PVS"
#define MODE_VHDL     "VHDL"
#define MODE_SMV      "SMV"
#define MODE_CONF     "CSP"
#define MODE_NETLIST  "Netlist"
#define MODE_SPECC    "SpecC"
#define MODE_PROMELA  "Promela"
#define MODE_PASCAL   "PASCAL"
#define MODE_CPP      "C++"
#define MODE_SIMPLIFY "Simplify"
#define MODE_BP       "bp"
#define MODE_CVC      "CVC"
#define MODE_CSHARP   "C#"
#define MODE_SMT      "SMT"
#define MODE_NSF      "NSF"
#define MODE_PHP      "PHP"
#define MODE_MDL      "MDL"

#endif
