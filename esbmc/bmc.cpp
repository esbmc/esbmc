/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Authors: Daniel Kroening, kroening@kroening.com
         Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sstream>
#include <fstream>

#include <i2string.h>
#include <location.h>
#include <time_stopping.h>
#include <message_stream.h>

#include <solvers/sat/satcheck.h>

#include <solvers/sat/dimacs_cnf.h>

#include <solvers/cvc/cvc_dec.h>

#include <solvers/boolector/boolector_dec.h>

#include <solvers/smt/smt_dec.h>

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
#include "version.h"

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

Function: bmc_baset::run_decision_procedure

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

decision_proceduret::resultt
bmc_baset::run_decision_procedure(prop_convt &prop_conv)
{
  static bool first_uw=false;
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
    }
    else
    {
      simple_slice(*equation);
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

    if(options.get_bool_option("minisat")) {
#ifdef MINISAT
      minisat_solver solver(*this);
      return solver.run_solver();
#else
      throw "This version of ESBMC was not compiled with minisat support";
#endif
    }

    if(options.get_bool_option("dimacs"))
      return write_dimacs();
    else if(options.get_bool_option("bl"))
#ifdef BOOLECTOR
    {
      boolector_solver solver(*this);
      return solver.run_solver();
    }
#else
      throw "This version of ESBMC was not compiled with boolector support";
#endif
    else if(options.get_bool_option("cvc"))
      return cvc();
    //else if(options.get_bool_option("smt"))
      //return smt();
    else if(options.get_bool_option("z3")) {
      z3_solver solver(*this);
      return solver.run_solver();
    }
    else if(options.get_bool_option("refine"))
      return bv_refinement();
    else
      throw "Please specify a SAT/SMT solver to use";
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

bool bmc_baset::solver_base::run_solver()
{

  switch(bmc.run_decision_procedure(*conv))
  {
  case decision_proceduret::D_UNSATISFIABLE:
    bmc.report_success();
    return false;

  case decision_proceduret::D_SATISFIABLE:
    bmc.error_trace(*conv);
    bmc.report_failure();
    return true;

  // XXX - and the use of this is...?
  case decision_proceduret::D_SMTLIB:
    return true;

  default:
    bmc.error("decision procedure failed");
    return true;
  }
}

#ifdef MINISAT
bmc_baset::minisat_solver::minisat_solver(bmc_baset &bmc)
  : solver_base(bmc), satcheck(), bv_cbmc(satcheck)
{
  satcheck.set_message_handler(bmc.message_handler);
  satcheck.set_verbosity(bmc.get_verbosity());

  if(bmc.options.get_option("arrays-uf")=="never")
    bv_cbmc.unbounded_array=bv_cbmct::U_NONE;
  else if(bmc.options.get_option("arrays-uf")=="always")
    bv_cbmc.unbounded_array=bv_cbmct::U_ALL;

  conv = &bv_cbmc;
}

bool bmc_baset::minisat_solver::run_solver()
{
  bool result = bmc_baset::solver_base::run_solver();

  if (result && bmc.options.get_bool_option("beautify-greedy"))
      counterexample_beautification_greedyt()(
        satcheck, bv_cbmc, *bmc.equation, bmc.symex.ns);

  return result;
}
#endif

bmc_baset::boolector_solver::boolector_solver(bmc_baset &bmc)
  : solver_base(bmc), boolector_dec()
{
  boolector_dec.set_file(bmc.options.get_option("outfile"));
  boolector_dec.set_btor(bmc.options.get_bool_option("btor"));
  conv = &boolector_dec;
}

bmc_baset::z3_solver::z3_solver(bmc_baset &bmc)
  : solver_base(bmc), z3_dec()
{
  z3_dec.set_encoding(bmc.options.get_bool_option("int-encoding"));
  z3_dec.set_file(bmc.options.get_option("outfile"));
  z3_dec.set_smt(bmc.options.get_bool_option("smt"));
  z3_dec.set_unsat_core(atol(bmc.options.get_option("core-size").c_str()));
  z3_dec.set_uw_models(bmc.options.get_bool_option("uw-model"));
  z3_dec.set_ecp(bmc.options.get_bool_option("ecp"));
  z3_dec.set_relevancy(bmc.options.get_bool_option("no-assume-guarantee"));
  conv = &z3_dec;
}

bool bmc_baset::z3_solver::run_solver()
{
  bool result = bmc_baset::solver_base::run_solver();
  bmc._unsat_core = z3_dec.get_z3_core_size();
  bmc._number_of_assumptions = z3_dec.get_z3_number_of_assumptions();
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
    if(options.get_bool_option("beautify-greedy"))
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

Function: bmc_baset::write_dimacs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::write_dimacs()
{
  const std::string &filename=options.get_option("outfile");
  
  if(filename.empty() || filename=="-")
    return write_dimacs(std::cout);

  std::ofstream out(filename.c_str());
  if(!out)
  {
    std::cerr << "failed to open " << filename << std::endl;
    return false;
  }

  return write_dimacs(out);
}

/*******************************************************************\

Function: bmc_baset::write_dimacs

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::write_dimacs(std::ostream &out)
{
  dimacs_cnft dimacs_cnf;
  dimacs_cnf.set_message_handler(message_handler);

  bv_cbmct bv_cbmc(dimacs_cnf);

  do_unwind_module(bv_cbmc);
  do_cbmc(bv_cbmc);

  bv_cbmc.dec_solve();

  dimacs_cnf.write_dimacs_cnf(out);
  
  return false;
}

/*******************************************************************\

Function: bmc_baset::cvc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::cvc()
{
  const std::string &filename=options.get_option("outfile");

  if(filename=="")
  {
    cvc_dect cvc_dec;
    return run_decision_procedure(cvc_dec);
  }
  else if(filename=="-")
    return cvc_conv(std::cout);
  else
  {
    std::ofstream out(filename.c_str());
    if(!out)
    {
      std::cerr << "failed to open " << filename << std::endl;
      return false;
    }

    return cvc_conv(out);
  }
}

/*******************************************************************\

Function: bmc_baset::cvc_conv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::cvc_conv(std::ostream &out)
{
  out << "%%%\n";
  out << "%%% Generated by ESBMC " << ESBMC_VERSION << "\n";
  out << "%%%\n\n";

  cvc_convt cvc_conv(out);
  cvc_conv.set_message_handler(message_handler);

  do_unwind_module(cvc_conv);
  do_cbmc(cvc_conv);

  cvc_conv.dec_solve();

  return false;
}

/*******************************************************************\

Function: bmc_baset::smt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::smt()
{
  const std::string &filename=options.get_option("outfile");

  if(filename=="")
  {
    smt_dect smt_dec;
    return run_decision_procedure(smt_dec);
  }
  else if(filename=="-")
    return smt_conv(std::cout);
  else
  {
    std::ofstream out(filename.c_str());
    if(!out)
    {
      std::cerr << "failed to open " << filename << std::endl;
      return false;
    }

    return smt_conv(out);
  }
}

/*******************************************************************\

Function: bmc_baset::smt_conv

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmc_baset::smt_conv(std::ostream &out)
{
  out << "%%%\n";
  out << "%%% Generated by ESBMC " << ESBMC_VERSION << "\n";
  out << "%%%\n\n";

  smt_convt smt_conv(out);
  smt_conv.set_message_handler(message_handler);

  do_unwind_module(smt_conv);
  do_cbmc(smt_conv);

  smt_conv.dec_solve();

  return false;
}
