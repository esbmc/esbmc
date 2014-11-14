/*******************************************************************\

Module: Traces of GOTO Programs

Author: Daniel Kroening

Date: July 2005

\*******************************************************************/

#ifndef CPROVER_GOTO_SYMEX_GOTO_TRACE_H
#define CPROVER_GOTO_SYMEX_GOTO_TRACE_H

#include <irep2.h>
#include <migrate.h>

#include <iostream>
#include <vector>

#include <fstream>
#include <map>
#include "VarMap.h"

#include <goto-programs/goto_program.h>

class goto_trace_stept 
{
public:
  unsigned step_nr;

  // See SSA_stept.
  std::vector<dstring> stack_trace;
  
  bool is_assignment() const { return type==ASSIGNMENT; }
  bool is_assume() const     { return type==ASSUME; }
  bool is_assert() const     { return type==ASSERT; }
  bool is_output() const     { return type==OUTPUT; }
  bool is_skip() const       { return type==SKIP; }
  bool is_renumber() const   { return type==RENUMBER; }

  typedef enum { ASSIGNMENT, ASSUME, ASSERT, OUTPUT, SKIP, RENUMBER } typet;
  typet type;
    
  goto_programt::const_targett pc;

  // this transition done by given thread number
  unsigned thread_nr;
  
  // for assume, assert, goto
  bool guard;
  
  // for assert
  std::string comment;

  // in SSA
  expr2tc lhs, rhs;
  
  // this is a constant
  expr2tc value;
  
  // original expression
  expr2tc original_lhs;

  // for OUTPUT
  std::string format_string;
  std::list<expr2tc> output_args;

  void output(
    const class namespacet &ns,
    std::ostream &out) const;
    
  goto_trace_stept():
    step_nr(0),
    thread_nr(0),
    guard(false)
  {
  }
};

class goto_tracet
{
public:
  typedef std::list<goto_trace_stept> stepst;
  typedef std::map< std::string, std::string, std::less< std::string > > mid;
  stepst steps;
  mid llvm_linemap;
  mid llvm_varmap;
  std::string FileName;
  std::string LineNumber;
  std::string FuncName;
  std::string VarName, OrigVarName;

  std::string mode, metadata_filename;
  
  void clear()
  {
    mode.clear();
    steps.clear();
  }

  void output(
    const class namespacet &ns,
    std::ostream &out) const;

  void open_llvm_varmap() {
	  std::ifstream inVarMap(metadata_filename.c_str(), std::ios::in);
	  if ( !inVarMap ) {
	     std::cerr << "Metadata File could not be opened." << std::endl;
	     exit( 1 );

      } // end if
	  VarMap vmap; // create record

	  // read first record from file
	  inVarMap.read( reinterpret_cast< char * >( &vmap ),
	     sizeof( VarMap ) );

	  // read all records from file
	  while ( inVarMap && !inVarMap.eof() ) {

	     // read record
	     if ( vmap.getLineNumber() != "" )
	    	llvm_linemap.insert( mid::value_type( vmap.getLineNumber() , vmap.getLineInfo() ) );
	        llvm_varmap.insert( mid::value_type( vmap.getVarName() , vmap.getVarInfo() ) );

	      // read next from file
	     inVarMap.read( reinterpret_cast< char * >( &vmap ),
	        sizeof( VarMap ) );

	   } // end while

  }

};

void get_metada_from_llvm(
  goto_tracet::stepst::const_iterator &it,
  const goto_tracet &goto_trace);

std::string get_varname_from_guard(
  goto_tracet::stepst::const_iterator &it,
  const goto_tracet &goto_trace);

void show_goto_trace_gui(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void show_goto_trace(
  std::ostream &out,
  const namespacet &ns,
  const goto_tracet &goto_trace);

void generate_goto_trace_in_graphml_format(
  std::string & tokenizer_path,
  std::string & filename,
  const namespacet & ns,
  const goto_tracet & goto_trace);

void counterexample_value(
  std::ostream &out,
  const namespacet &ns,
  const expr2tc &identifier,
  const expr2tc &value);

#endif
