/*******************************************************************\

Module: SMT-LIB Frontend, parse tree

Author: CM Wintersteiger

\*******************************************************************/

#ifndef SMT_PARSE_TREET_H_
#define SMT_PARSE_TREET_H_

#include <expr.h>
#include <hash_cont.h>

class smt_parse_treet
{
public:  
  
  class benchmarkt {
    public:
      enum status_idt { SAT, UNSAT, UNKNOWN };
  
      typedef std::list<irept> logicst;
      typedef std::list<exprt> assumptionst;
      typedef std::list<exprt> formulast;
      typedef std::list<status_idt> statussest;
      typedef hash_map_cont<irep_idt, std::list<exprt>, irep_id_hash > 
        function_symbolst;
      typedef hash_map_cont<irep_idt, std::list<exprt>, irep_id_hash > 
        predicate_symbolst;
      typedef std::list<exprt> sort_symbolst;
      typedef std::list<irep_idt> notest;
      typedef std::map<irep_idt, irep_idt> annotationst;
      
      irept name;
      logicst logics;
      statussest status;
      notest notes;
      annotationst annotations;
      
      function_symbolst function_symbols;
      predicate_symbolst predicate_symbols;
      sort_symbolst sort_symbols;
      assumptionst assumptions;
      formulast formulas;
      locationt location;
  };
  
  class theoryt {
    public:
      typedef std::list<exprt> sortst;
      typedef hash_map_cont<irep_idt, std::list<exprt>, irep_id_hash > 
        funst;
      typedef hash_map_cont<irep_idt, std::list<exprt>, irep_id_hash > 
        predst;
      typedef std::list<exprt> axiomst;
      typedef std::list<irep_idt> notest;
      typedef std::list<irep_idt> definitionst;
      typedef std::map<irep_idt, irep_idt> annotationst;
      
      irept name;
      sortst sorts;
      funst funs;
      predst preds;
      definitionst definitions;
      axiomst axioms;
      notest notes;
      annotationst annotations;
      locationt location;
  };
  
  class logict {
    public:
      typedef std::list<irept> theoriest;
      typedef std::list<irep_idt> languagest;
      typedef std::list<irep_idt> extensionst;
      typedef std::list<irep_idt> notest;
      typedef std::map<irep_idt, irep_idt> annotationst;
      
      irept name;
      theoriest theories;
      languagest languages;
      extensionst extensions;
      notest notes;
      annotationst annotations;
      locationt location;
  };
  
  typedef std::list<benchmarkt> benchmarkst;
  typedef std::list<theoryt> theoriest;
  typedef std::list<logict> logicst;
  
  benchmarkst benchmarks;
  theoriest theories;
  logicst logics;

  void swap(smt_parse_treet &smt_parse_tree);
  void clear();
};


#endif /*SMT_PARSE_TREET_H_*/
