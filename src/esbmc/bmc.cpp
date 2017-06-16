/*******************************************************************\

Module: Symbolic Execution of ANSI-C

Authors: Daniel Kroening, kroening@kroening.com
         Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <csignal>
#include <sys/types.h>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#include <winbase.h>
#undef ERROR
#undef small
#endif

#include <ac_config.h>
#include <esbmc/bmc.h>
#include <esbmc/document_subgoals.h>
#include <fstream>
#include <goto-symex/build_goto_trace.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/reachability_tree.h>
#include <goto-symex/slice.h>
#include <goto-symex/xml_goto_trace.h>
#include <langapi/language_util.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <sstream>
#include <util/i2string.h>
#include <util/irep2.h>
#include <util/location.h>
#include <util/message_stream.h>
#include <util/migrate.h>
#include <util/time_stopping.h>

bmct::bmct(const goto_functionst &funcs,
  optionst &opts,
  contextt &_context,
  message_handlert &_message_handler)
  : messaget(_message_handler),
    options(opts),
    context(_context),
    ns(context),
    ui(ui_message_handlert::PLAIN)
{
  interleaving_number = 0;
  interleaving_failed = 0;

  ltl_results_seen[ltl_res_bad] = 0;
  ltl_results_seen[ltl_res_failing] = 0;
  ltl_results_seen[ltl_res_succeeding] = 0;
  ltl_results_seen[ltl_res_good] = 0;

  if(options.get_bool_option("smt-during-symex"))
  {
    runtime_solver =
      boost::shared_ptr<smt_convt>(
        create_solver_factory(
          "",
          opts.get_bool_option("int-encoding"),
          ns,
          options));

    symex =
      std::make_shared<reachability_treet>(
        funcs, ns, options,
        boost::shared_ptr<runtime_encoded_equationt>(
          new runtime_encoded_equationt(ns, *runtime_solver)),
        _context,
        _message_handler);
  }
  else
  {
    symex =
      std::make_shared<reachability_treet>(
        funcs,
        ns,
        options,
        boost::shared_ptr<symex_target_equationt>(
          new symex_target_equationt(ns)),
        _context,
        _message_handler);
  }
}


void bmct::do_cbmc(
  boost::shared_ptr<smt_convt> &smt_conv,
  boost::shared_ptr<symex_target_equationt> &eq)
{
  smt_conv->set_message_handler(message_handler);
  eq->convert(*smt_conv.get());
}

void bmct::successful_trace(boost::shared_ptr<symex_target_equationt> &eq)
{
  if(options.get_bool_option("result-only"))
    return;

  goto_tracet goto_trace;
  std::string witness_output = options.get_option("witness-output");
  int specification = 0;
  if(!witness_output.empty())
    set_ui(ui_message_handlert::GRAPHML);

  switch(ui)
  {
    case ui_message_handlert::GRAPHML:
      status("Building successful trace");
      build_successful_goto_trace(eq, ns, goto_trace);
      specification += options.get_bool_option("overflow-check") ? 1 : 0;
      specification += options.get_bool_option("memory-leak-check") ? 2 : 0;
      generate_goto_trace_in_correctness_graphml_format(
        witness_output,
        options.get_bool_option("witness-detailed"),
        specification,
        ns,
        goto_trace
      );
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

void bmct::error_trace(
  boost::shared_ptr<smt_convt> &smt_conv,
  boost::shared_ptr<symex_target_equationt> &eq)
{
  if(options.get_bool_option("result-only"))
    return;

  status("Building error trace");

  goto_tracet goto_trace;
  int specification = 0;
  build_goto_trace(eq, smt_conv, goto_trace);

  std::string witness_output = options.get_option("witness-output");
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
        witness_output,
        options.get_bool_option("witness-detailed"),
        specification,
        ns,
        goto_trace
      );

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
bmct::run_decision_procedure(
  boost::shared_ptr<smt_convt> &smt_conv,
  boost::shared_ptr<symex_target_equationt> &eq)
{
  std::string logic;

  if (!options.get_bool_option("int-encoding"))
  {
    logic = "bit-vector";
    logic += (!config.ansi_c.use_fixed_for_float) ? "/floating-point " : " ";
    logic += "arithmetic";
  }
  else
    logic = "integer/real arithmetic";

  std::cout << "Encoding remaining VCC(s) using " << logic << "\n";

  smt_conv->set_message_handler(message_handler);
  smt_conv->set_verbosity(get_verbosity());

  fine_timet encode_start = current_time();
  do_cbmc(smt_conv, eq);
  fine_timet encode_stop = current_time();

  std::ostringstream str;
  str << "Encoding to solver time: ";
  output_time(encode_stop - encode_start, str);
  str << "s";
  status(str.str());

  if(options.get_bool_option("smt-formula-too")
     || options.get_bool_option("smt-formula-only"))
  {
    smt_conv->dump_smt();
    if(options.get_bool_option("smt-formula-only")) return smt_convt::P_SMTLIB;
  }

  std::stringstream ss;
  ss << "Solving with solver " << smt_conv->solver_text();
  status(ss.str());

  fine_timet sat_start=current_time();
  smt_convt::resultt dec_result = smt_conv->dec_solve();
  fine_timet sat_stop=current_time();

  // output runtime
  str.clear();
  str << "\nRuntime decision procedure: ";
  output_time(sat_stop-sat_start, str);
  str << "s";
  status(str.str());

  return dec_result;
}

void bmct::report_success()
{
  status("\nVERIFICATION SUCCESSFUL");

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

void bmct::report_failure()
{
  status("\nVERIFICATION FAILED");

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

void bmct::show_program(boost::shared_ptr<symex_target_equationt> &eq)
{
  unsigned count=1;

  languagest languages(ns, MODE_C);

  std::cout << "\n" << "Program constraints: " << eq->SSA_steps.size() << "\n";

  bool print_guard = config.options.get_bool_option("show-guards");
  bool sparse = config.options.get_bool_option("simple-ssa-printing");

  for(auto const &it : eq->SSA_steps)
  {
    if(!(it.is_assert() || it.is_assignment() || it.is_assume()))
      continue;

    if (!sparse) {
      std::cout << "// " << it.source.pc->location_number << " ";
      std::cout << it.source.pc->location.as_string() << "\n";
    }

    std::cout <<   "(" << count << ") ";

    std::string string_value;

    exprt cond = migrate_expr_back(it.cond);
    languages.from_expr(cond, string_value);

    if(it.is_assignment())
    {
      std::cout << string_value << "\n";
    }
    else if(it.is_assert())
    {
      std::cout << "(assert)" << string_value << "\n";
    }
    else if(it.is_assume())
    {
      std::cout << "(assume)" << string_value << "\n";
    }
    else if (it.is_renumber())
    {
      std::cout << "renumber: " << from_expr(ns, "", it.lhs) << "\n";
    }

    if(!migrate_expr_back(it.guard).is_true() && print_guard)
    {
      languages.from_expr(migrate_expr_back(it.guard), string_value);
      std::cout << std::string(i2string(count).size()+3, ' ');
      std::cout << "guard: " << string_value << "\n";
    }

    if (!sparse) {
      std::cout << "\n";
    }

    count++;
  }
}

smt_convt::resultt bmct::start_bmc()
{
  boost::shared_ptr<symex_target_equationt> eq;

  auto res = run(eq);
  switch(res)
  {
    case smt_convt::P_UNSATISFIABLE:
      if(!options.get_bool_option("base-case"))
      {
        successful_trace(eq);
        report_success();
      }
      else
        status("No bug has been found in the base case");
      break;

    case smt_convt::P_SATISFIABLE:
      if (!options.get_bool_option("base-case") &&
          options.get_bool_option("show-counter-example"))
      {
        error_trace(runtime_solver, eq);
      }
      else if(!options.get_bool_option("inductive-step")
          && !options.get_bool_option("forward-condition"))
      {
        error_trace(runtime_solver, eq);
        report_failure();
      }
      else if (options.get_bool_option("forward-condition"))
        status("The forward condition is unable to prove the property");
      else
        status("The inductive step is unable to prove the property");
      break;

    // Return failure if we didn't actually check anything, we just emitted the
    // test information to an SMTLIB formatted file. Causes esbmc to quit
    // immediately (with no error reported)
    case smt_convt::P_SMTLIB:
      break;

    default:
      error("decision procedure failed");
      break;
  }

  return res;
}

smt_convt::resultt bmct::run(boost::shared_ptr<symex_target_equationt> &eq)
{
  symex->options.set_option("unwind", options.get_option("unwind"));
  symex->setup_for_new_explore();

  if(options.get_bool_option("schedule"))
    return run_thread(eq);

  do
  {
    if(++interleaving_number > 1)
    {
      std::cout << "*** Thread interleavings " << interleaving_number
                << " ***" << std::endl;
    }

    fine_timet bmc_start = current_time();
    smt_convt::resultt res = run_thread(eq);
    if(res)
    {
      ++interleaving_failed;
      if(!options.get_bool_option("all-runs"))
        return res;
    }
    fine_timet bmc_stop = current_time();

    std::ostringstream str;
    str << "BMC program time: ";
    output_time(bmc_stop-bmc_start, str);
    str << "s";
    status(str.str());

    // Only run for one run
    if (options.get_bool_option("interactive-ileaves"))
      return smt_convt::P_UNSATISFIABLE;

  } while(symex->setup_next_formula());

  if(options.get_bool_option("all-runs"))
  {
    std::cout << "*** number of generated interleavings: " << interleaving_number << " ***" << std::endl;
    std::cout << "*** number of failed interleavings: " << interleaving_failed << " ***" << std::endl;
  }

  if (options.get_bool_option("ltl")) {
    // So, what was the lowest value ltl outcome that we saw?
    if (ltl_results_seen[ltl_res_bad]) {
      std::cout << "Final lowest outcome: LTL_BAD" << std::endl;
      return smt_convt::P_UNSATISFIABLE;
    } else if (ltl_results_seen[ltl_res_failing]) {
      std::cout << "Final lowest outcome: LTL_FAILING" << std::endl;
      return smt_convt::P_UNSATISFIABLE;
    } else if (ltl_results_seen[ltl_res_succeeding]) {
      std::cout << "Final lowest outcome: LTL_SUCCEEDING" << std::endl;
      return smt_convt::P_UNSATISFIABLE;
    } else if (ltl_results_seen[ltl_res_good]) {
      std::cout << "Final lowest outcome: LTL_GOOD" << std::endl;
      return smt_convt::P_UNSATISFIABLE;
    } else {
      std::cout << "No traces seen, apparently" << std::endl;
      return smt_convt::P_UNSATISFIABLE;
    }
  }

  return smt_convt::P_UNSATISFIABLE;
}

smt_convt::resultt bmct::run_thread(boost::shared_ptr<symex_target_equationt> &eq)
{
  boost::shared_ptr<goto_symext::symex_resultt> result;

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
    return smt_convt::P_ERROR;
  }

  catch(const char *error_str)
  {
    message_streamt message_stream(*get_message_handler());
    message_stream.error(error_str);
    return smt_convt::P_ERROR;
  }

  catch(std::bad_alloc&)
  {
    std::cout << "Out of memory" << std::endl;
    return smt_convt::P_ERROR;
  }

  fine_timet symex_stop = current_time();

  eq = boost::dynamic_pointer_cast<symex_target_equationt>(result->target);

  {
    std::ostringstream str;
    str << "Symex completed in: ";
    output_time(symex_stop - symex_start, str);
    str << "s";
    str << " (" << eq->SSA_steps.size() << " assignments)";
    status(str.str());
  }

  if (options.get_bool_option("double-assign-check"))
    eq->check_for_duplicate_assigns();

  try
  {
    fine_timet slice_start = current_time();
    u_int64_t ignored;
    if(!options.get_bool_option("no-slice"))
    {
      ignored = slice(eq);
    }
    else
    {
      ignored = simple_slice(eq);
    }
    fine_timet slice_stop = current_time();

    {
      std::ostringstream str;
      str << "Slicing time: ";
      output_time(slice_stop - slice_start, str);
      str << "s";
      str << " (removed " << ignored << " assignments)";
      status(str.str());
    }

    if (options.get_bool_option("program-only") ||
        options.get_bool_option("program-too"))
      show_program(eq);

    if (options.get_bool_option("program-only"))
      return smt_convt::P_SMTLIB;

    {
      std::ostringstream str;
      str << "Generated " << result->total_claims << " VCC(s), ";
      str << result->remaining_claims << " remaining after simplification ";
      str << "(" << eq->SSA_steps.size() - ignored << " assignments)";
      status(str.str());
    }

    if(options.get_bool_option("document-subgoals"))
    {
      document_subgoals(*eq.get(), std::cout);
      return smt_convt::P_SMTLIB;
    }

    if(options.get_bool_option("show-vcc"))
    {
      show_vcc(eq);
      return smt_convt::P_SMTLIB;
    }

    if(result->remaining_claims==0)
      return smt_convt::P_UNSATISFIABLE;

    if (options.get_bool_option("ltl")) {
      int res = ltl_run_thread(eq);
      // Record that we've seen this outcome; later decide what the least
      // outcome was.
      ltl_results_seen[res]++;
      return smt_convt::P_UNSATISFIABLE;
    }

    if (!options.get_bool_option("smt-during-symex")) {
      runtime_solver =
        boost::shared_ptr<smt_convt>(
          create_solver_factory(
            "",
            options.get_bool_option("int-encoding"),
            ns,
            options));
    }

    return run_decision_procedure(runtime_solver, eq);
  }

  catch(std::string &error_str)
  {
    error(error_str);
    return smt_convt::P_ERROR;
  }

  catch(const char *error_str)
  {
    error(error_str);
    return smt_convt::P_ERROR;
  }

  catch(std::bad_alloc&)
  {
    std::cout << "Out of memory" << std::endl;
    return smt_convt::P_ERROR;
  }
}

int
bmct::ltl_run_thread(boost::shared_ptr<symex_target_equationt> &equation)
{
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
    if (run(equation)) {
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
    if (run(equation)) {
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
    if (run(equation)) {
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
