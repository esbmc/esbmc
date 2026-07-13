#include <ac_config.h>

#include <esbmc/bmc.h>
#include <esbmc/esbmc_parseoptions.h>
#include <goto-symex/goto_symex.h>
#include <goto-symex/goto_trace.h>
#include <goto-symex/sarif.h>
#include <util/cwe_mapping.h>
#include <solvers/smt/smt_result.h>
#include <solvers/smtlib/smtlib_conv.h>
#include <solvers/solve.h>
#include <cctype>
#include <charconv>
#include <clang-c-frontend/clang_c_language.h>
#include <util/config.h>
#include <util/filesystem.h>
#include <csignal>
#include <cstdlib>
#include <limits>
#include <util/expr_util.h>
#include <iostream>
#include <fstream>
#include <goto-programs/add_race_assertions.h>
#include <goto-programs/add_restrict_assertions.h>
#include <goto-programs/goto_atomicity_check.h>
#include <goto-programs/goto_check.h>
#include <goto-programs/goto_convert_functions.h>
#include <goto-programs/goto_inline.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_termination.h>
#include <esbmc/ranking_synthesis.h>
#include <esbmc/non_termination.h>
#include <goto-programs/goto_loop_simplify.h>
#include <goto-programs/goto_loop_invariant.h>
#include <goto-programs/abstract-interpretation/interval_analysis.h>
#include <goto-programs/abstract-interpretation/gcse.h>
#include <goto-programs/loop_numbers.h>
#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/read_cbmc_goto_object.h>
#include <goto-programs/write_goto_binary.h>
#include <goto-programs/remove_no_op.h>
#include <goto-programs/remove_unreachable.h>
#include <goto-programs/remove_exceptions.h>
#include <goto-programs/set_claims.h>
#include <goto-programs/show_claims.h>
#include <goto-programs/loop_unroll.h>
#include <goto-programs/goto_check_uninit_vars.h>
#include <goto-programs/goto_check_unchecked_return.h>
#include <goto-programs/dead_store_analysis.h>
#include <goto-programs/mark_decl_as_non_det.h>
#include <goto-programs/assign_params_as_non_det.h>
#include <goto2c/goto2c.h>
#include <util/irep.h>
#include <langapi/languages.h>
#include <langapi/mode.h>
#include <memory>
#include <pointer-analysis/goto_program_dereference.h>
#include <pointer-analysis/show_value_sets.h>
#include <pointer-analysis/value_set_analysis.h>
#include <util/symbol.h>
#include <util/time_stopping.h>
#include <goto-programs/goto_cfg.h>
#include <langapi/language_util.h>
#include <goto-programs/contracts/contracts.h>

#ifndef _WIN32
#  include <sys/wait.h>
#  include <fcntl.h>
#  ifdef __GLIBC__
#    include <execinfo.h>
#  endif
#endif

#ifdef ENABLE_GOTO_CONTRACTOR
#  include <goto-programs/goto_contractor.h>
#endif

bool esbmc_parseoptionst::set_claims(goto_functionst &goto_functions)
{
  try
  {
    if (cmdline.isset("claim"))
      ::set_claims(goto_functions, cmdline.get_values("claim"));
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (int)
  {
    return true;
  }

  return false;
}

// This method performs a wide range of actions that can be broadly divided
// into 3 main steps:
//
//  1) creating a GOTO program,
//  2) processing the GOTO program, and
//  3) outputting the GOTO program.
//
// This method is typically used as the second stage
// (right after parsing the command line options) by the verification methods
// (i.e., BMC, k-induction, etc).
//
// \param options - various options used during the above steps,
// \param goto_functions - the "created and processed" GOTO program.
bool esbmc_parseoptionst::get_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    fine_timet create_start = current_time();
    if (create_goto_program(options, goto_functions))
      return true;
    fine_timet create_stop = current_time();
    log_status(
      "GOTO program creation time: {}s",
      time2string(create_stop - create_start));

    fine_timet process_start = current_time();
    if (process_goto_program(options, goto_functions))
      return true;
    fine_timet process_stop = current_time();
    log_status(
      "GOTO program processing time: {}s",
      time2string(process_stop - process_start));
    if (output_goto_program(options, goto_functions))
      return true;
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

// Retarget the synthesised __ESBMC_main wrapper so its boilerplate call to
// c:@F@main instead dispatches to `target`. Used to bridge the entry point of a
// loaded --binary onto the program's real entry: a CBMC goto-binary's
// __CPROVER__start, or a user-selected --function harness. Without this,
// __ESBMC_main would run the empty boilerplate main and report a verdict over
// essentially no program. No-op if __ESBMC_main was not synthesised.
static void
retarget_esbmc_main(goto_functionst &goto_functions, const irep_idt &target)
{
  auto entry = goto_functions.function_map.find("__ESBMC_main");
  if (entry == goto_functions.function_map.end())
    return;

  Forall_goto_program_instructions (it, entry->second.body)
  {
    if (!it->is_function_call())
      continue;

    code_function_call2t &call = to_code_function_call2t(it->code);
    if (
      is_symbol2t(call.function) &&
      to_symbol2t(call.function).thename == "c:@F@main")
      call.function = symbol2tc(get_empty_type(), target);
  }
}

// This method creates a GOTO program from the source specified by the
// command line options. A GOTO program can be created:
//
//  1) from a GOTO binary file,
//  2) by parsing the input program files.
//
// \param options - options to be passed through,
// \param goto_functions - this is where the created GOTO program is stored.
static void link_cbmc_libc_bodies(goto_functionst &goto_functions);

bool esbmc_parseoptionst::create_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    if (cmdline.args.size() == 0)
    {
      log_error("Please provide a program to verify");
      return true;
    }

    // If the user is providing the GOTO functions, we don't need to parse
    if (cmdline.isset("binary"))
    {
      if (cmdline.isset("cprover"))
        log_warning(
          "--cprover is deprecated and has no effect; ESBMC additions are now "
          "linked automatically for CBMC goto-binaries (disable with "
          "--no-cprover-additions)");

      // A CBMC goto-binary needs ESBMC's additions (the __ESBMC_main entry
      // wrapper and the CPROVER-intrinsic bodies). Synthesise and link them
      // automatically, before reading the binaries so that goto_convert only
      // ever runs over the boilerplate and never clobbers the loaded bodies.
      const bool cbmc_additions =
        has_cbmc_binary_input() && !cmdline.isset("no-cprover-additions");
      if (cbmc_additions)
      {
        log_status(
          "CBMC goto-binary detected: linking ESBMC additions automatically");
        if (synthesize_cprover_additions(options, goto_functions))
          return true;
      }

      if (read_goto_binary(goto_functions))
        return true;

      // Resolve CBMC's bodyless libc externals (ceil/floor/..., strlen/strcmp/
      // strncmp) to the operational-model bodies the additions linked, before
      // symex sees a bodyless call returning nondet.
      if (cbmc_additions)
        link_cbmc_libc_bodies(goto_functions);

      // Bridge the synthesised __ESBMC_main, which wraps the boilerplate
      // c:@F@main, onto the program's real entry. An explicit --function wins;
      // otherwise a CBMC binary dispatches into __CPROVER__start (it runs
      // __CPROVER_initialize and calls the program's main/harness). Without
      // this, a CBMC binary verifies the empty boilerplate main and may report
      // a spurious SUCCESSFUL.
      if (cmdline.isset("function"))
        retarget_esbmc_main(goto_functions, cmdline.getval("function"));
      else if (
        cbmc_additions && goto_functions.function_map.count("__CPROVER__start"))
        retarget_esbmc_main(goto_functions, "__CPROVER__start");
      else if (cbmc_additions)
        log_warning(
          "CBMC goto-binary support is experimental: no entry point to bridge "
          "(no __CPROVER__start and no --function), so __ESBMC_main wraps the "
          "boilerplate main and the verdict may be unsound.");

      goto_functions.update();
    }
    else
    {
      if (parse_goto_program(options, goto_functions))
        return true;
    }
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

// This method creates a GOTO program from the given GOTO binary.
//
// \param goto_functions - this is where the created GOTO program is stored.
bool esbmc_parseoptionst::read_goto_binary(goto_functionst &goto_functions)
{
  log_progress("Reading GOTO program from file");
  goto_binary_reader goto_reader;
  for (const auto &arg : cmdline.args)
  {
    if (goto_reader.read_goto_binary(arg, context, goto_functions))
    {
      log_error("Failed to open `{}'", arg);
      return true;
    }
  }

  return false;
}

// True if any --binary input is a CBMC goto-binary. CBMC's format starts with
// the magic 0x7f 'G' 'B' 'F'; ESBMC's own format starts with 'G' 'B' 'F'.
bool esbmc_parseoptionst::has_cbmc_binary_input()
{
  for (const auto &arg : cmdline.args)
  {
    std::ifstream in(arg, std::ios::in | std::ios::binary);
    unsigned char hdr[4] = {0, 0, 0, 0};
    // Bounded read: capped at sizeof(hdr) into the fixed-size buffer, and the
    // gcount() check confirms a full header before any byte is inspected
    // (CWE-120/CWE-20).
    in.read(reinterpret_cast<char *>(hdr), sizeof(hdr)); // Flawfinder: ignore
    if (
      in.gcount() >= static_cast<std::streamsize>(sizeof(hdr)) &&
      is_cbmc_goto_magic(hdr))
      return true;
  }
  return false;
}

// Bridge CBMC's plain-named bodyless libc externals (e.g. `ceil`, `strlen`) to
// the operational-model bodies the additions link under the C-frontend-mangled
// id (`c:@F@ceil`). CBMC emits libm (ceil/floor/trunc/round, float/long-double
// variants) and string.h (strlen/strcmp/strncmp) functions as bodyless
// FUNCTION_CALL externals; unlike sqrt/fabs -- rewritten to expressions in
// cbmc_adapter -- they have no ESBMC expression form and must run the library
// body. Copying the bodied function's body and type onto the bodyless
// declaration lets symex resolve the call: argument_assignments binds actual
// args using the copied type's parameter names, which match the copied body
// (goto-symex/symex_function.cpp). The string bodies are byte loops, so a call
// with a symbolic length needs an `--unwind` bound like any other loop.
static void link_cbmc_libc_bodies(goto_functionst &goto_functions)
{
  static const char *const libc[] = {
    "ceilf",     "ceil",     "ceill",     "floorf", "floor",  "floorl",
    "truncf",    "trunc",    "truncl",    "roundf", "round",  "roundl",
    "copysignf", "copysign", "copysignl", "fminf",  "fmin",   "fminl",
    "fmaxf",     "fmax",     "fmaxl",     "fdimf",  "fdim",   "fdiml",
    "modff",     "modf",     "modfl",     "strlen", "strcmp", "strncmp"};

  for (const char *name : libc)
  {
    auto bodyless = goto_functions.function_map.find(name);
    if (
      bodyless == goto_functions.function_map.end() ||
      bodyless->second.body_available)
      continue;

    auto bodied = goto_functions.function_map.find(std::string("c:@F@") + name);
    if (
      bodied == goto_functions.function_map.end() ||
      !bodied->second.body_available)
      continue;

    bodyless->second.body = bodied->second.body; // operator= fixes up targets
    bodyless->second.type = bodied->second.type;
    bodyless->second.body_available = true;
  }
}

// Compile a boilerplate translation unit through the normal C-frontend pipeline
// to obtain ESBMC's "additions": typecheck() pulls in the C library via
// add_cprover_library, final() builds the __ESBMC_main entry wrapper, and
// goto_convert() turns them into goto bodies. This is exactly the prebuilt
// library.goto that one otherwise links manually. We run it before reading the
// CBMC binary so goto_convert only ever sees the boilerplate's symbols.
bool esbmc_parseoptionst::synthesize_cprover_additions(
  optionst &options,
  goto_functionst &goto_functions)
{
  file_operations::tmp_file tf =
    file_operations::create_tmp_file("esbmc-cprover-%%%%-%%%%-%%%%.c");
  // Taking the addresses of the bodied libc functions marks them referenced, so
  // add_cprover_library links their operational-model bodies into the additions;
  // link_cbmc_libc_bodies then bridges the CBMC binary's plain-named bodyless
  // declarations to them. Unlike sqrt/fabs (operators rewritten in cbmc_adapter)
  // these have no ESBMC expression form and must run the C library body -- that
  // includes the string.h query functions (strlen/strcmp/strncmp), whose bodies
  // are byte loops. Referencing memcpy/memmove/memset additionally force-links
  // string.c, whose bodies pull in __memcpy_impl/__memmove_impl/__memset_impl --
  // the byte-loop fallbacks intrinsic_memcpy/memmove/memset bump to when a copy's
  // size or pointers are symbolic (and, for memmove, when the regions overlap).
  // The cbmc_adapter retargets CBMC's memcpy/memset/memmove calls straight to the
  // c:@F@__ESBMC_* intrinsics, but those intrinsics still need the *_impl bodies
  // present for the bump path, so the boilerplate must link them here.
  static const char boilerplate[] =
    "/* Auto-generated: bundle all ESBMC additions for CBMC gotos. */\n"
    "#include <math.h>\n"
    "#include <string.h>\n"
    "void *const __esbmc_cbmc_libc_refs[] = {\n"
    "  (void *)ceilf,     (void *)ceil,     (void *)ceill,\n"
    "  (void *)floorf,    (void *)floor,    (void *)floorl,\n"
    "  (void *)truncf,    (void *)trunc,    (void *)truncl,\n"
    "  (void *)roundf,    (void *)round,    (void *)roundl,\n"
    "  (void *)copysignf, (void *)copysign, (void *)copysignl,\n"
    "  (void *)fminf,     (void *)fmin,     (void *)fminl,\n"
    "  (void *)fmaxf,     (void *)fmax,     (void *)fmaxl,\n"
    "  (void *)fdimf,     (void *)fdim,     (void *)fdiml,\n"
    "  (void *)modff,     (void *)modf,     (void *)modfl,\n"
    "  (void *)memcpy,    (void *)memmove,  (void *)memset,\n"
    "  (void *)strlen,    (void *)strcmp,   (void *)strncmp,\n"
    "};\n"
    "int main(void) { return 0; }\n";
  if (fputs(boilerplate, tf.file()) == EOF || fflush(tf.file()) != 0)
  {
    log_error("could not write boilerplate for CPROVER additions");
    return true;
  }

  // Point the command line at the boilerplate for this one compile, then
  // restore it for the subsequent binary read. The user's --function names a
  // harness in the CBMC binary, not in this boilerplate, so also neutralise
  // config.main (which config.cpp sets from --function) for the compile --
  // otherwise the boilerplate's entry-point synthesis (clang_c_main) looks for
  // that harness here and aborts with "main symbol not found". The retarget in
  // create_goto_program applies --function to the loaded binary afterwards.
  cmdlinet::argst saved_args = cmdline.args;
  const std::string saved_main = config.main;
  cmdline.args = {tf.path()};
  config.main = "";
  bool failed = parse_goto_program(options, goto_functions);
  cmdline.args = saved_args;
  config.main = saved_main;

  if (failed)
    log_error("failed to synthesize ESBMC additions for CBMC goto-binary");
  return failed;
}

// This method creates a GOTO program by parsing the input program files.
//
// \param options - options to be passed to the program parser,
// \param goto_functions - this is where the created GOTO program is stored.
bool esbmc_parseoptionst::parse_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    if (parse(cmdline))
      return true;

    if (cmdline.isset("parse-tree-too") || cmdline.isset("parse-tree-only"))
    {
      std::ostringstream oss;
      for (auto &it : langmap)
        it.second->show_parse(oss);
      log_status("{}", oss.str());
      if (cmdline.isset("parse-tree-only"))
        exit(0);
    }

    // Typechecking (old frontend) or adjust (clang frontend)
    if (typecheck())
      return true;
    if (final())
      return true;

    // we no longer need any parse trees or language files
    clear_parse();

    if (cmdline.isset("symbol-table-too") || cmdline.isset("symbol-table-only"))
    {
      std::ostringstream oss;
      show_symbol_table_plain(oss);
      log_status("{}", oss.str());
      if (cmdline.isset("symbol-table-only"))
        exit(0);
    }

    // Expand --no-standard-checks into individual options before goto_convert,
    // because VLA size checks are generated during goto conversion.
    if (
      cmdline.isset("no-standard-checks") ||
      options.get_bool_option("no-standard-checks"))
    {
      options.set_option("no-pointer-check", true);
      options.set_option("no-div-by-zero-check", true);
      options.set_option("no-pointer-relation-check", true);
      options.set_option("no-unlimited-scanf-check", true);
      options.set_option("no-vla-size-check", true);
      options.set_option("no-align-check", true);
      options.set_option("no-bounds-check", true);
    }

    log_progress("Generating GOTO Program");
    goto_convert(context, options, goto_functions);
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
    return true;
  }

  return false;
}

// This method provides different output methods for the given GOTO program.
// Depending on the provided options this method can:
//
//  - output the given GOTO program as text,
//  - translate the provided GOTO program into C,
//  - create a GOTO binary from this GOTO program,
//  - methods outputting some additional information of the GOTO program.
//
// \param options - various options setting the output methods,
// \param goto_functions - the GOTO program to be output.
bool esbmc_parseoptionst::output_goto_program(
  optionst &options,
  goto_functionst &goto_functions)
{
  try
  {
    namespacet ns(context);

    // show it?
    if (cmdline.isset("show-loops"))
    {
      show_loop_numbers(goto_functions);
      return true;
    }

    // show it?
    if (cmdline.isset("show-goto-value-sets"))
    {
      value_set_analysist value_set_analysis(ns);
      value_set_analysis(goto_functions);
      std::ostringstream oss;
      show_value_sets(goto_functions, value_set_analysis, oss);
      log_result("{}", oss.str());
      return true;
    }

    // Write the GOTO program into a binary
    if (cmdline.isset("output-goto"))
    {
      log_status("Writing GOTO program to file");
      std::ofstream oss(
        cmdline.getval("output-goto"), std::ios::out | std::ios::binary);
      if (write_goto_binary(oss, context, goto_functions))
      {
        log_error("Failed to generate goto binary file"); // TODO: explain why
        abort();
      };
      return true;
    }

    if (cmdline.isset("show-ileave-points"))
    {
      print_ileave_points(ns, goto_functions);
      return true;
    }

    // Output the GOTO program to the log (and terminate or continue) in
    // a human-readable format
    if (
      cmdline.isset("goto-functions-too") ||
      cmdline.isset("goto-functions-only"))
    {
      std::ostringstream oss;
      goto_functions.output(ns, oss);
      log_status("{}", oss.str());
      if (cmdline.isset("goto-functions-only"))
        exit(0);
    }

    if (cmdline.isset("dump-goto-cfg"))
    {
      goto_cfg cfg(goto_functions);
      cfg.dump_graph();
      return true;
    }

    // Print a flat list of every function call site with its arguments.
    // Output format: caller -> callee(arg1, arg2, ...) [file:line]
    // Nested calls appear as separate lines with compiler-generated
    // temporaries (e.g. return_value$_add$5) showing data flow.
    if (cmdline.isset("show-call-sites"))
    {
      for (const auto &f : goto_functions.function_map)
      {
        if (!f.second.body_available)
          continue;
        const std::string caller = f.first.as_string();
        forall_goto_program_instructions (i_it, f.second.body)
        {
          if (i_it->is_function_call())
          {
            const auto &fc = to_code_function_call2t(i_it->code);

            // Direct calls have a symbol; indirect calls (function
            // pointers) fall back to pretty-printing the expression.
            std::string callee;
            if (is_symbol2t(fc.function))
              callee = to_symbol2t(fc.function).get_symbol_name();
            else
              callee = from_expr(ns, "", fc.function);

            // Pretty-print each actual argument as a comma-separated list
            std::string args;
            for (size_t i = 0; i < fc.operands.size(); i++)
            {
              if (i > 0)
                args += ", ";
              args += from_expr(ns, "", fc.operands[i]);
            }

            std::string loc;
            if (!i_it->location.get_file().empty())
            {
              const auto &file = i_it->location.get_file();
              const auto &line = i_it->location.get_line();

              if (!line.empty())
                loc = " [" + file.as_string() + ":" + line.as_string() + "]";
              else
                loc = " [" + file.as_string() + "]";
            }
            log_status("{} -> {}({}){}", caller, callee, args, loc);
          }
        }
      }
      std::exit(0);
    }

    // Translate the GOTO program to C and output it into the log or
    // a specified output file
    if (cmdline.isset("goto2c"))
    {
      // Creating a translator here
      goto2ct goto2c(ns, goto_functions);
      goto2c.preprocess();
      goto2c.check();
      std::string res = goto2c.translate();

      const std::string &filename = options.get_option("output");
      if (!filename.empty())
      {
        // Outputting the translated program into the output file
        std::ofstream out(filename);
        assert(out);
        out << res;
      }
      else
        std::cout << res;
      return true;
    }
  }

  catch (const char *e)
  {
    log_error("{}", e);
    return true;
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
    return true;
  }

  return false;
}

// This performs the preprocessing of the input program
// when the old C/C++ frontend (i.e., from "ansi-c/" or "cpp/") is used.
void esbmc_parseoptionst::preprocessing()
{
  try
  {
    if (cmdline.args.size() != 1)
    {
      log_error("Please provide one program to preprocess");
      return;
    }

    std::string filename = cmdline.args[0];

    // To test that the file exists,
    std::ifstream infile(filename.c_str());
    if (!infile)
    {
      log_error("failed to open input file");
      return;
    }
#ifdef ENABLE_OLD_FRONTEND
    std::ostringstream oss;
    if (c_preprocess(filename, oss, false))
      log_error("PREPROCESSING ERROR");
    log_status("{}", oss.str());
#endif
  }
  catch (const char *e)
  {
    log_error("{}", e);
  }

  catch (const std::string &e)
  {
    log_error("{}", e);
  }

  catch (std::bad_alloc &)
  {
    log_error("Out of memory");
  }
}
