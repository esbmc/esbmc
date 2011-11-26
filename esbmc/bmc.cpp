/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

extern "C" {
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>

#include <sys/sendfile.h>
}

#include <sstream>
#include <fstream>

#include <i2string.h>
#include <location.h>
#include <time_stopping.h>
#include <message_stream.h>

#include <solvers/sat/satcheck.h>

#include <langapi/mode.h>
#include <langapi/languages.h>
#include <langapi/language_util.h>

#include <goto-symex/goto_trace.h>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/slice.h>
#include <goto-symex/slice_by_trace.h>
#include <goto-symex/xml_goto_trace.h>

#ifdef HAVE_BV_REFINEMENT
#include <bv_refinement/bv_refinement_loop.h>
#endif

#include "bmc.h"
#include "bv_cbmc.h"
#include "counterexample_beautification_greedy.h"
#include "document_subgoals.h"

/*******************************************************************\

Function: bmc_baset::do_unwind_module

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::do_unwind_module(
  decision_proceduret &decision_procedure)
{
}

/*******************************************************************\

Function: bmc_baset::do_cbmc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::do_cbmc(prop_convt &solver)
{
  solver.set_message_handler(message_handler);

  equation->convert(solver);

  forall_expr_list(it, bmc_constraints)
    solver.set_to_true(*it);
}

/*******************************************************************\

Function: bmc_baset::error_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::error_trace(const prop_convt &prop_conv)
{
  status("Building error trace");

  goto_tracet goto_trace;
  build_goto_trace(*equation, prop_conv, goto_trace);

  switch(ui)
  {
  case ui_message_handlert::PLAIN:
    std::cout << std::endl << "Counterexample:" << std::endl;
    show_goto_trace(std::cout, symex.ns, goto_trace);
    break;

  case ui_message_handlert::OLD_GUI:
    show_goto_trace_gui(std::cout, symex.ns, goto_trace);
    break;

  case ui_message_handlert::XML_UI:
    {
      xmlt xml;
      convert(symex.ns, goto_trace, xml);
      std::cout << xml << std::endl;
    }
    break;

  default:
    assert(false);
  }
}

/*******************************************************************\

Function: bmc_baset::decide_default

  Inputs:

 Outputs:

 Purpose: Decide using "default" decision procedure

\*******************************************************************/

bool bmc_baset::decide_default()
{
  sat_minimizert satcheck;
  satcheck.set_message_handler(message_handler);
  satcheck.set_verbosity(get_verbosity());

  bv_cbmct bv_cbmc(satcheck);

  if(options.get_option("arrays-uf")=="never")
    bv_cbmc.unbounded_array=bv_cbmct::U_NONE;
  else if(options.get_option("arrays-uf")=="always")
    bv_cbmc.unbounded_array=bv_cbmct::U_ALL;

  bool result=true;

  switch(run_decision_procedure(bv_cbmc))
  {
  case decision_proceduret::D_UNSATISFIABLE:
    result=false;
    report_success();
    break;

  case decision_proceduret::D_SATISFIABLE:
    if(options.get_bool_option("beautify-pbs"))
      throw "beautify-pbs is no longer supported";
    else if(options.get_bool_option("beautify-greedy"))
      counterexample_beautification_greedyt()(
        satcheck, bv_cbmc, *equation, symex.ns);

    error_trace(bv_cbmc);
    report_failure();
    break;

  default:
    error("decision procedure failed");
  }

  return result;
}

/*******************************************************************\

Function: bmc_baset::decide_solver_boolector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::decide_solver_boolector()
{
  bool result=true;
  boolector_dect boolector_dec;

  boolector_dec.set_file(options.get_option("outfile"));
  boolector_dec.set_btor(options.get_bool_option("btor"));

  switch(run_decision_procedure(boolector_dec))
  {
    case decision_proceduret::D_UNSATISFIABLE:
      result=false;
      report_success();
      break;

    case decision_proceduret::D_SATISFIABLE:
	  error_trace(boolector_dec);
      report_failure();
      break;

    case decision_proceduret::D_SMTLIB:
      break;

    default:
      error("decision procedure failed");
  }

  return result;
}

/*******************************************************************\

Function: bmc_baset::decide_solver_z3

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::decide_solver_z3()
{
  bool result=true;
  z3_dect z3_dec;

  z3_dec.set_encoding(options.get_bool_option("int-encoding"));
  z3_dec.set_file(options.get_option("outfile"));
  z3_dec.set_smt(options.get_bool_option("smt"));
  z3_dec.set_unsat_core(atol(options.get_option("core-size").c_str()));
  z3_dec.set_uw_models(options.get_bool_option("uw-model"));
  z3_dec.set_ecp(options.get_bool_option("ecp"));
  z3_dec.set_relevancy(options.get_bool_option("no-assume-guarantee"));

  switch(run_decision_procedure(z3_dec))
  {
    case decision_proceduret::D_UNSATISFIABLE:
      result=false;
      report_success();
      break;
    case decision_proceduret::D_SATISFIABLE:
      result=true;
      if (!options.get_bool_option("ecp"))
      {
	    error_trace(z3_dec);
        report_failure();
      }
      break;
    case decision_proceduret::D_SMTLIB:
      break;
    default:
      error("decision procedure failed");
  }

  _unsat_core = z3_dec.get_z3_core_size();
  _number_of_assumptions = z3_dec.get_z3_number_of_assumptions();

  //std::cout << "_unsat_core: " << _unsat_core << std::endl;
  //std::cout << "result: " << result << std::endl;

  return result;
}

/*******************************************************************\

Function: bmc_baset::bv_refinement

  Inputs:

 Outputs:

 Purpose: Decide using refinement decision procedure

\*******************************************************************/

bool bmc_baset::bv_refinement()
{
  #ifdef HAVE_BV_REFINEMENT
  satcheckt satcheck;
  satcheck.set_message_handler(message_handler);
  satcheck.set_verbosity(get_verbosity());

  bv_refinement_loopt bv_refinement_loop(satcheck);
  bv_refinement_loop.set_message_handler(message_handler);
  bv_refinement_loop.set_verbosity(get_verbosity());

  bool result=true;

  switch(run_decision_procedure(bv_refinement_loop))
  {
  case decision_proceduret::D_UNSATISFIABLE:
    result=false;
    report_success();
    break;

  case decision_proceduret::D_SATISFIABLE:
    if(options.get_bool_option("beautify-pbs"))
      throw "beautify-pbs is no longer supported";
    else if(options.get_bool_option("beautify-greedy"))
      throw "refinement doesn't support greedy beautification";

    error_trace(bv_refinement_loop);
    report_failure();
    break;

  default:
    error("decision procedure failed");
  }

  return result;
  #else
  throw "bv refinement not linked in";
  #endif
}

/*******************************************************************\

Function: bmc_baset::run_decision_procedure

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt
bmc_baset::run_decision_procedure(prop_convt &prop_conv)
{
  static bool first_uw=false;
//  status("Passing problem to "+prop_conv.decision_procedure_text());
  std::string logic;
  if (options.get_bool_option("bl-bv") || options.get_bool_option("z3-bv") ||
	  options.get_bool_option("bl") || !options.get_bool_option("int-encoding"))
	logic = "bit-vector arithmetic";
  else
	logic = "integer/real arithmetic";

  if (!(options.get_bool_option("minisat")) && !options.get_bool_option("smt")
		  && !options.get_bool_option("btor"))
    std::cout << "Encoding remaining VCC(s) using " << logic << "\n";
  prop_conv.set_message_handler(message_handler);
  prop_conv.set_verbosity(get_verbosity());

  // stop the time
  fine_timet sat_start=current_time();

  do_unwind_module(prop_conv);
  do_cbmc(prop_conv);

//  status("Running "+prop_conv.decision_procedure_text());

  decision_proceduret::resultt dec_result=prop_conv.dec_solve();

  // output runtime
  if (!options.get_bool_option("smt") && !options.get_bool_option("btor"))
  {
    std::ostringstream str;
    fine_timet sat_stop=current_time();
    str << "Runtime decision procedure: ";
    output_time(sat_stop-sat_start, str);
    str << "s";
    status(str.str());
  }

  if(symex.options.get_bool_option("uw-model") && first_uw)
  {
    std::cout << "number of assumptions: " << _number_of_assumptions << " literal(s)"<< std::endl;
    std::cout << "size of the unsatisfiable core: " << _unsat_core << " literal(s)"<< std::endl;
  }
  else
	first_uw=true;

  return dec_result;
}

/*******************************************************************\

Function: bmc_baset::report_success

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::report_success()
{
  status("VERIFICATION SUCCESSFUL");

  switch(ui)
  {
  case ui_message_handlert::OLD_GUI:
    std::cout << "SUCCESS" << std::endl
              << "Verification successful" << std::endl
              << ""     << std::endl
              << ""     << std::endl
              << ""     << std::endl
              << ""     << std::endl;
    break;

  case ui_message_handlert::PLAIN:
    break;

  case ui_message_handlert::XML_UI:
    {
      xmlt xml("cprover-status");
      xml.data="SUCCESS";
      std::cout << xml;
      std::cout << std::endl;
    }
    break;

  default:
    assert(false);
  }

}

/*******************************************************************\

Function: bmc_baset::report_failure

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::report_failure()
{
  status("VERIFICATION FAILED");

  switch(ui)
  {
  case ui_message_handlert::OLD_GUI:
    break;

  case ui_message_handlert::PLAIN:
    break;

  case ui_message_handlert::XML_UI:
    {
      xmlt xml("cprover-status");
      xml.data="FAILURE";
      std::cout << xml;
      std::cout << std::endl;
    }
    break;

  default:
    assert(false);
  }
}

/*******************************************************************\

Function: bmc_baset::show_program

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::show_program()
{
  unsigned count=1;

  languagest languages(symex.ns, MODE_C);

  std::cout << std::endl << "Program constraints:" << std::endl;

  for(symex_target_equationt::SSA_stepst::const_iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if(it->is_assignment())
    {
      std::string string_value;
      languages.from_expr(it->cond, string_value);
      std::cout << "(" << count << ") " << string_value << std::endl;
      count++;
    }
#if 1
    else if(it->is_assert())
    {
      std::string string_value;
      languages.from_expr(it->cond, string_value);
      std::cout << "(" << count << ") " << "(assert)" << string_value << std::endl;
      count++;
    }
#endif
  }
}

/*******************************************************************\

Function: bmc_baset::run

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::run(const goto_functionst &goto_functions)
{
  //symex.total_claims=0;
  static bool resp;
  static unsigned int interleaving_number=0, interleaving_failed=0, uw_loop=0;
  symex.set_message_handler(message_handler);
  symex.set_verbosity(get_verbosity());
  symex.options=options;

  symex.last_location.make_nil();

  // get unwinding info
  setup_unwind();

  if(symex.options.get_bool_option("schedule"))
  {
	if(symex.options.get_bool_option("uw-model"))
	  std::cout << "*** UW loop " << ++uw_loop << " ***" << std::endl;

	resp = run_thread(goto_functions);


	//underapproximation-widening model
	while (_unsat_core)
	{
	  equation->clear();
	  symex.total_claims=0;
	  symex.remaining_claims=0;
	  std::cout << "*** UW loop " << ++uw_loop << " ***" << std::endl;
	  resp = run_thread(goto_functions);
	}

    return resp;
  }
  else
  {
    symex.multi_formulas_init(goto_functions);

    while(symex.multi_formulas_has_more_formula())
    {
  	  equation->clear();
  	  symex.total_claims=0;
  	  symex.remaining_claims=0;

      if (++interleaving_number>1)
        std::cout << "*** Thread interleavings " << interleaving_number << " ***" << std::endl;

      if(run_thread(goto_functions))
      {
    	++interleaving_failed;
        if(!symex.options.get_bool_option("all-runs"))
        {
          return true;
        }
      }
    }
  }

  if (symex.options.get_bool_option("all-runs"))
  {
    std::cout << "*** number of generated interleavings: " << interleaving_number << " ***" << std::endl;
    std::cout << "*** number of failed interleavings: " << interleaving_failed << " ***" << std::endl;
  }

  std::cout << "ohai:" << std::endl;
  int fd = open("/proc/self/status", O_RDONLY, 0);
  sendfile(STDOUT_FILENO, fd, NULL, 4096);
  close(fd);

  return false;
}

bool bmc_baset::run_thread(const goto_functionst &goto_functions)
{
  try
  {
    if(symex.options.get_bool_option("schedule"))
    {
      symex(goto_functions);
    }
    else
    {
      symex.multi_formulas_get_next_formula();
      equation = &symex.art1->_cur_target_state->_target;
    }
  }

  catch(std::string &error_str)
  {
    message_streamt message_stream(*get_message_handler());
    message_stream.err_location(symex.last_location);
    message_stream.error(error_str);
    return true;
  }

  catch(const char *error_str)
  {
    message_streamt message_stream(*get_message_handler());
    message_stream.err_location(symex.last_location);
    message_stream.error(error_str);
    return true;
  }

  catch(std::bad_alloc)
  {
    message_streamt message_stream(*get_message_handler());
    message_stream.error("Out of memory");
    return true;
  }

  print(8, "size of program expression: "+
           i2string(equation->SSA_steps.size())+
           " assignments");

  try
  {
    if(options.get_option("slice-by-trace")!="")
    {
      symex_slice_by_tracet symex_slice_by_trace;
      symex_slice_by_trace.slice_by_trace
	(options.get_option("slice-by-trace"), *equation, symex.ns);
    }

    if(options.get_bool_option("slice-formula"))
    {
      slice(*equation);
#if 0
      print(8, "slicing removed "+
        i2string(equation.count_ignored_SSA_steps())+" assignments");
#endif
    }
    else
    {
      simple_slice(*equation);
#if 0
      print(8, "simple slicing removed "+
        i2string(equation.count_ignored_SSA_steps())+" assignments");
#endif
    }

    if(options.get_bool_option("program-only"))
    {
      show_program();
      return false;
    }

    {
      std::string msg;
      msg="Generated "+i2string(symex.total_claims)+
          " VCC(s), "+i2string(symex.remaining_claims)+
          " remaining after simplification";
      print(8, msg);
    }

    if(options.get_bool_option("document-subgoals"))
    {
      document_subgoals(*equation, std::cout);
      return false;
    }

    if(options.get_bool_option("show-vcc"))
    {
      show_vcc();
      return false;
    }

    if(symex.remaining_claims==0)
    {
      report_success();
      return false;
    }

    if(options.get_bool_option("minisat"))
      return decide_default();
    if(options.get_bool_option("dimacs"))
      return write_dimacs();
    else if(options.get_bool_option("bl"))
      return boolector();
    else if(options.get_bool_option("cvc"))
      return cvc();
    //else if(options.get_bool_option("smt"))
      //return smt();
    else if(options.get_bool_option("z3"))
      return z3();
    else if(options.get_bool_option("refine"))
      return bv_refinement();
    else
      return decide_default();
  }

  catch(std::string &error_str)
  {
    error(error_str);
    return true;
  }

  catch(const char *error_str)
  {
    error(error_str);
    return true;
  }

  catch(std::bad_alloc)
  {
    error("Out of memory");
    abort();
    //return true; jmorse
  }
}

/*******************************************************************\

Function: bmc_baset::setup_unwind

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmc_baset::setup_unwind()
{
  const std::string &set = options.get_option("unwindset");
  unsigned int length = set.length();

  for(unsigned int idx = 0; idx < length; idx++)
  {
    std::string::size_type next = set.find(",", idx);
    std::string val = set.substr(idx, next - idx);
    unsigned long id = atoi(val.substr(0, val.find(":", 0)).c_str());
    unsigned long uw = atol(val.substr(val.find(":", 0) + 1).c_str());
    symex.unwind_set[id] = uw;
    if(next == std::string::npos) break;
    idx = next;
  }

  symex.max_unwind=atol(options.get_option("unwind").c_str());
}
