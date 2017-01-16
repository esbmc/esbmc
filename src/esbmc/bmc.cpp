/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Authors: Daniel Kroening, kroening@kroening.com
         Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sys/types.h>

#include <signal.h>
#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <winbase.h>
#undef ERROR
#undef small
#endif

#include <sstream>
#include <fstream>

#include <irep2.h>
#include <i2string.h>
#include <location.h>
#include <time_stopping.h>
#include <message_stream.h>
#include <migrate.h>

#include <langapi/mode.h>
#include <langapi/languages.h>
#include <langapi/language_util.h>

#include <goto-symex/goto_trace.h>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/slice.h>
#include <goto-symex/xml_goto_trace.h>
#include <goto-symex/reachability_tree.h>

#include "bmc.h"
#include "document_subgoals.h"
#include <ac_config.h>

static volatile bool checkpoint_sig = false;

void
sigusr1_handler(int sig __attribute__((unused)))
{

  checkpoint_sig = true;
  return;
}

/*******************************************************************\

Function: bmct::do_cbmc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::do_cbmc(smt_convt &solver, symex_target_equationt &equation)
{
  solver.set_message_handler(message_handler);

  equation.convert(solver);
}

/*******************************************************************\

Function: bmct::successful_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::successful_trace(symex_target_equationt &equation __attribute__((unused)))
{
  if(options.get_bool_option("base-case"))
  {
    status("No bug has been found in the base case");
    return ;
  }

  if(options.get_bool_option("result-only"))
    return;

  goto_tracet goto_trace;
  std::string witness_output = options.get_option("witness-output");
  std::string programfile = options.get_option("witness-programfile");
  int specification = 0;
  if(!witness_output.empty())
    set_ui(ui_message_handlert::GRAPHML);

  switch(ui)
  {
    case ui_message_handlert::GRAPHML:
      status("Building successful trace");
      build_successful_goto_trace(equation, ns, goto_trace);
      specification += options.get_bool_option("overflow-check") ? 1 : 0;
      specification += options.get_bool_option("memory-leak-check") ? 2 : 0;
      generate_goto_trace_in_correctness_graphml_format(
        programfile,
        witness_output,
        options.get_bool_option("witness-detailed"),
        specification,
        ns,
        goto_trace
      );
      std::cout << "The correctness witness in GraphML format is available at: "
                << options.get_option("witness-output")
                << std::endl;
    break;

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

Function: bmct::error_trace

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::error_trace(smt_convt &smt_conv,
                       symex_target_equationt &equation)
{
  if(options.get_bool_option("result-only"))
    return;

  status("Building error trace");

  goto_tracet goto_trace;
  int specification = 0;
  build_goto_trace(equation, smt_conv, goto_trace);

  std::string witness_output = options.get_option("witness-output");
  std::string programfile = options.get_option("witness-programfile");
  if(!witness_output.empty())
  {
    set_ui(ui_message_handlert::GRAPHML);
  }

  switch (ui)
  {
    case ui_message_handlert::GRAPHML:
      specification += options.get_bool_option("overflow-check") ? 1 : 0;
      specification += options.get_bool_option("memory-leak-check") ? 2 : 0;
      generate_goto_trace_in_violation_graphml_format(
        programfile,
        witness_output,
        options.get_bool_option("witness-detailed"),
        specification,
        ns,
        goto_trace
      );
      std::cout
        << "The violation witness in GraphML format is available at: "
        << options.get_option("witness-output")
        << std::endl;
      std::cout << std::endl << "Counterexample:" << std::endl;
      show_goto_trace(std::cout, ns, goto_trace);
    break;

    case ui_message_handlert::PLAIN:
      std::cout << std::endl << "Counterexample:" << std::endl;
      show_goto_trace(std::cout, ns, goto_trace);
    break;

    case ui_message_handlert::OLD_GUI:
      show_goto_trace_gui(std::cout, ns, goto_trace);
    break;

    case ui_message_handlert::XML_UI:
    {
      xmlt xml;
      convert(ns, goto_trace, xml);
      std::cout << xml << std::endl;
      break;
    }

    default:
      assert(false);
  }
}

smt_convt::resultt
bmct::run_decision_procedure(smt_convt &smt_conv,
                             symex_target_equationt &equation)
{
  std::string logic;

  if (!options.get_bool_option("int-encoding"))
    logic = "bit-vector arithmetic";
  else
    logic = "integer/real arithmetic";

  if (!options.get_bool_option("smt"))
    std::cout << "Encoding remaining VCC(s) using " << logic << "\n";

  smt_conv.set_message_handler(message_handler);
  smt_conv.set_verbosity(get_verbosity());

  fine_timet encode_start = current_time();
  do_cbmc(smt_conv, equation);
  fine_timet encode_stop = current_time();

  if (!options.get_bool_option("smt"))
  {
    std::ostringstream str;
    str << "Encoding to solver time: ";
    output_time(encode_stop - encode_start, str);
    str << "s";
    status(str.str());
  }

  if(options.get_bool_option("dump-smt-formula"))
    smt_conv.dump_SMT();

  std::stringstream ss;
  ss << "Solving with solver " << smt_conv.solver_text();
  status(ss.str());

  fine_timet sat_start=current_time();
  smt_convt::resultt dec_result=smt_conv.dec_solve();
  fine_timet sat_stop=current_time();

  // output runtime
  if (!options.get_bool_option("smt"))
  {
    std::ostringstream str;
    str << "Runtime decision procedure: ";
    output_time(sat_stop-sat_start, str);
    str << "s";
    status(str.str());
  }

  return dec_result;
}

/*******************************************************************\

Function: bmct::report_success

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::report_success()
{

  if(options.get_bool_option("base-case"))
  {
    status("No bug has been found in the base case");
    return ;
  }

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

    case ui_message_handlert::GRAPHML:
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

Function: bmct::report_failure

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::report_failure()
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

  case ui_message_handlert::GRAPHML:
  break;

  default:
    assert(false);
  }
}

/*******************************************************************\

Function: bmct::show_program

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void bmct::show_program(symex_target_equationt &equation)
{
  unsigned count=1;

  languagest languages(ns, MODE_C);

  std::cout << "\n" << "Program constraints:" << "\n";

  bool print_guard = config.options.get_bool_option("dump-guards");
  bool sparse = config.options.get_bool_option("simple-ssa-printing");

  for(symex_target_equationt::SSA_stepst::const_iterator
      it=equation.SSA_steps.begin();
      it!=equation.SSA_steps.end(); it++)
  {
    if (!sparse) {
      std::cout << "// " << it->source.pc->location_number << " ";
      std::cout << it->source.pc->location.as_string() << "\n";
    }

    std::cout <<   "(" << count << ") ";

    std::string string_value;

    if(it->is_assignment())
    {
      languages.from_expr(migrate_expr_back(it->cond), string_value);
      std::cout << string_value << "\n";
    }
    else if(it->is_assert())
    {
      languages.from_expr(migrate_expr_back(it->cond), string_value);
      std::cout << "(assert)" << string_value << "\n";
    }
    else if(it->is_assume())
    {
      languages.from_expr(migrate_expr_back(it->cond), string_value);
      std::cout << "(assume)" << string_value << "\n";
    }
    else if (it->is_renumber())
    {
      std::cout << "renumber: " << from_expr(ns, "", it->lhs) << "\n";
    }

    if(!migrate_expr_back(it->guard).is_true() && print_guard)
    {
      languages.from_expr(migrate_expr_back(it->guard), string_value);
      std::cout << std::string(i2string(count).size()+3, ' ');
      std::cout << "guard: " << string_value << "\n";
    }

    if (!sparse) {
      std::cout << "\n";
    }

    count++;
  }
}

/*******************************************************************\

Function: bmct::run

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool bmct::run(void)
{
#ifndef _WIN32
  struct sigaction act;
#endif
  bool resp;

#ifndef _WIN32
  // Collect SIGUSR1, indicating that we're supposed to checkpoint.
  act.sa_handler = sigusr1_handler;
  sigemptyset(&act.sa_mask);
  act.sa_flags = 0;
  sigaction(SIGUSR1, &act, NULL);
#endif

  symex->options.set_option("unwind", options.get_option("unwind"));
  symex->setup_for_new_explore();

  if(options.get_bool_option("schedule"))
  {
    resp = run_thread();
    return resp;
  }
  else
  {
    if (options.get_bool_option("from-checkpoint")) {
      if (options.get_option("checkpoint-file") == "") {
        std::cerr << "Please provide a checkpoint file" << std::endl;
        abort();
      }

      reachability_treet::dfs_position pos(
                                         options.get_option("checkpoint-file"));
      symex->restore_from_dfs_state((void*)&pos);
    }

    do
    {
      if(!options.get_bool_option("k-induction")
        && !options.get_bool_option("k-induction-parallel"))
        if (++interleaving_number>1) {
          print(8, "*** Thread interleavings "+
            i2string((unsigned long)interleaving_number)+
            " ***");
        }

      fine_timet bmc_start = current_time();
      if(run_thread())
      {
        ++interleaving_failed;

        if (options.get_bool_option("checkpoint-on-cex")) {
          write_checkpoint();
        }

        if(!options.get_bool_option("all-runs"))
        {
          return true;
        }
      }
      fine_timet bmc_stop = current_time();

      std::ostringstream str;
      str << "BMC program time: ";
      output_time(bmc_stop-bmc_start, str);
      str << "s";
      status(str.str());

      if (checkpoint_sig) {
        write_checkpoint();
      }

      // Only run for one run
      if (options.get_bool_option("interactive-ileaves"))
        return false;

    } while(symex->setup_next_formula());
  }

  if (options.get_bool_option("all-runs"))
  {
    std::cout << "*** number of generated interleavings: " << interleaving_number << " ***" << std::endl;
    std::cout << "*** number of failed interleavings: " << interleaving_failed << " ***" << std::endl;
  }

  if (options.get_bool_option("ltl")) {
    // So, what was the lowest value ltl outcome that we saw?
    if (ltl_results_seen[ltl_res_bad]) {
      std::cout << "Final lowest outcome: LTL_BAD" << std::endl;
      return false;
    } else if (ltl_results_seen[ltl_res_failing]) {
      std::cout << "Final lowest outcome: LTL_FAILING" << std::endl;
      return false;
    } else if (ltl_results_seen[ltl_res_succeeding]) {
      std::cout << "Final lowest outcome: LTL_SUCCEEDING" << std::endl;
      return false;
    } else if (ltl_results_seen[ltl_res_good]) {
      std::cout << "Final lowest outcome: LTL_GOOD" << std::endl;
      return false;
    } else {
      std::cout << "No traces seen, apparently" << std::endl;
      return false;
    }
  }

  return false;
}

bool bmct::run_thread()
{
  std::shared_ptr<goto_symext::symex_resultt> result;
  bool ret;

  fine_timet symex_start = current_time();
  try
  {
    if(options.get_bool_option("schedule"))
    {
      result = symex->generate_schedule_formula();
    }
    else
    {
      result = symex->get_next_formula();
    }
  }

  catch(std::string &error_str)
  {
    message_streamt message_stream(*get_message_handler());
    message_stream.error(error_str);
    return true;
  }

  catch(const char *error_str)
  {
    message_streamt message_stream(*get_message_handler());
    message_stream.error(error_str);
    return true;
  }

  catch(std::bad_alloc&)
  {
    std::cout << "Out of memory" << std::endl;
    return true;
  }

  fine_timet symex_stop = current_time();

  std::ostringstream str;
  str << "Symex completed in: ";
  output_time(symex_stop - symex_start, str);
  str << "s";
  status(str.str());

  auto equation =
    std::dynamic_pointer_cast<symex_target_equationt>(result->target);

  print(8, "size of program expression: "+
           i2string((unsigned long)equation.get()->SSA_steps.size())+
           " assignments");

  if (options.get_bool_option("double-assign-check")) {
    equation.get()->check_for_duplicate_assigns();
  }

  try
  {

    fine_timet slice_start = current_time();
    if(!options.get_bool_option("no-slice"))
    {
      slice(*equation);
    }
    else
    {
      simple_slice(*equation);
    }
    fine_timet slice_stop = current_time();

    std::ostringstream str;
    str << "Slicing time: ";
    output_time(slice_stop - slice_start, str);
    str << "s";
    status(str.str());

    if (options.get_bool_option("program-only") ||
        options.get_bool_option("program-too"))
      show_program(*equation);

    if (options.get_bool_option("program-only"))
      return false;

    {
      std::string msg;
      msg="Generated "+i2string(result->total_claims)+
          " VCC(s), "+i2string(result->remaining_claims)+
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
      show_vcc(*equation);
      return false;
    }

    if(result->remaining_claims==0)
    {
      successful_trace(*equation);
      report_success();
      return false;
    }

    if (options.get_bool_option("ltl")) {
      int res = ltl_run_thread(equation.get());
      // Record that we've seen this outcome; later decide what the least
      // outcome was.
      ltl_results_seen[res]++;
      return false;
    }

    if (options.get_bool_option("smt"))
      if (interleaving_number !=
          (unsigned int) strtol(options.get_option("smtlib-ileave-num").c_str(), NULL, 10))
        return false;

    if (!options.get_bool_option("smt-during-symex")) {
      runtime_solver = create_solver_factory("",
                                             options.get_bool_option("int-encoding"),
                                             ns, options);
    }

    ret = run_solver(*equation, runtime_solver);

    return ret;
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

  catch(std::bad_alloc&)
  {
    std::cout << "Out of memory" << std::endl;
    return true;
  }
}

int
bmct::ltl_run_thread(symex_target_equationt *equation __attribute__((unused)))
{
  smt_convt *solver;
  bool ret;
  unsigned int num_asserts = 0;
  // LTL checking - first check for whether we have an indeterminate prefix,
  // and then check for all others.

  // Start by turning all assertions that aren't the negative prefix
  // assertion into skips.
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if (it->is_assert()) {
      if (it->comment != "LTL_BAD") {
        it->type = goto_trace_stept::SKIP;
      } else {
        num_asserts++;
      }
    }
  }

  std::cout << "Checking for LTL_BAD" << std::endl;
  if (num_asserts != 0) {
    solver = create_solver_factory("z3",
                                   options.get_bool_option("int-encoding"),
                                   ns, options);
    ret = run_solver(*equation, solver);
    delete solver;
    if (ret) {
      std::cout << "Found trace satisfying LTL_BAD" << std::endl;
      return ltl_res_bad;
    }
  } else {
    std::cerr << "Warning: Couldn't find LTL_BAD assertion" << std::endl;
  }

  // Didn't find it; turn skip steps back into assertions.
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if (it->type == goto_trace_stept::SKIP)
      it->type = goto_trace_stept::ASSERT;
  }

  // Try again, with LTL_FAILING
  num_asserts = 0;
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if (it->is_assert()) {
      if (it->comment != "LTL_FAILING") {
        it->type = goto_trace_stept::SKIP;
      } else {
        num_asserts++;
      }
    }
  }

  std::cout << "Checking for LTL_FAILING" << std::endl;
  if (num_asserts != 0) {
    solver = create_solver_factory("z3",
                                   options.get_bool_option("int-encoding"),
                                   ns, options);
    ret = run_solver(*equation, solver);
    delete solver;
    if (ret) {
      std::cout << "Found trace satisfying LTL_FAILING" << std::endl;
      return ltl_res_failing;
    }
  } else {
    std::cerr << "Warning: Couldn't find LTL_FAILING assertion" <<std::endl;
  }

  // Didn't find it; turn skip steps back into assertions.
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if (it->type == goto_trace_stept::SKIP)
      it->type = goto_trace_stept::ASSERT;
  }

  // Try again, with LTL_SUCCEEDING
  num_asserts = 0;
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if (it->is_assert()) {
      if (it->comment != "LTL_SUCCEEDING") {
        it->type = goto_trace_stept::SKIP;
      } else {
        num_asserts++;
      }
    }
  }

  std::cout << "Checking for LTL_SUCCEEDING" << std::endl;
  if (num_asserts != 0) {
    solver = create_solver_factory("z3",
                                   options.get_bool_option("int-encoding"),
                                   ns, options);
    ret = run_solver(*equation, solver);
    delete solver;
    if (ret) {
      std::cout << "Found trace satisfying LTL_SUCCEEDING" << std::endl;
      return ltl_res_succeeding;
    }
  } else {
    std::cerr << "Warning: Couldn't find LTL_SUCCEEDING assertion"
              << std::endl;
  }

  // Otherwise, we just got a good prefix.
  for(symex_target_equationt::SSA_stepst::iterator
      it=equation->SSA_steps.begin();
      it!=equation->SSA_steps.end(); it++)
  {
    if (it->type == goto_trace_stept::SKIP)
      it->type = goto_trace_stept::ASSERT;
  }

  return ltl_res_good;
}

bool bmct::run_solver(symex_target_equationt &equation, smt_convt *solver)
{

  switch(run_decision_procedure(*solver, equation))
  {
    case smt_convt::P_UNSATISFIABLE:
      if(!options.get_bool_option("base-case"))
      {
        successful_trace(equation);
        report_success();
      }
      else
        status("No bug has been found in the base case");
      return false;

    case smt_convt::P_SATISFIABLE:
      if (!options.get_bool_option("base-case") &&
          options.get_bool_option("show-counter-example"))
      {
        error_trace(*solver, equation);
      }
      else if(!options.get_bool_option("inductive-step")
    		  && !options.get_bool_option("forward-condition"))
      {
        error_trace(*solver, equation);
   	    report_failure();
      }
      else if (options.get_bool_option("forward-condition"))
        status("The forward condition is unable to prove the property");
      else
        status("The inductive step is unable to prove the property");

      return true;

    // Return failure if we didn't actually check anything, we just emitted the
    // test information to an SMTLIB formatted file. Causes esbmc to quit
    // immediately (with no error reported)
    case smt_convt::P_SMTLIB:
      return true;

    default:
      error("decision procedure failed");
      return true;
  }
}

void bmct::write_checkpoint(void)
{
  std::string f;

  if (options.get_option("checkpoint-file") == "") {
    char buffer[32];
#ifndef _WIN32
    pid_t pid = getpid();
#else
    unsigned long pid = GetCurrentProcessId();
#endif
    sprintf(buffer, "%d", pid);
    f = "esbmc_checkpoint." + std::string(buffer);
  } else {
    f = options.get_option("checkpoint-file");
  }

  symex->save_checkpoint(f);
  return;
}
