#ifndef CPROVER_ESBMC_PARSEOPTIONS_H
#define CPROVER_ESBMC_PARSEOPTIONS_H

#include <esbmc/bmc.h>
#include <goto-programs/goto_convert_functions.h>
#include <langapi/language_ui.h>
#include <util/cmdline.h>
#include <util/options.h>
#include <util/parseoptions.h>
#include <util/algorithms.h>

extern const struct group_opt_templ all_cmd_options[];

class esbmc_parseoptionst : public parseoptions_baset, public language_uit
{
public:
  int doit() override;
  void help() override;

  esbmc_parseoptionst(int argc, const char **argv)
    : parseoptions_baset(all_cmd_options, argc, argv), language_uit(cmdline)
  {
  }

  ~esbmc_parseoptionst()
  {
    close_file(out);
    if(out != err)
      close_file(err);
  }

protected:
  virtual void get_command_line_options(optionst &options);
  virtual int do_bmc(bmct &bmc);

  virtual bool
  get_goto_program(optionst &options, goto_functionst &goto_functions);

  virtual bool
  process_goto_program(optionst &options, goto_functionst &goto_functions);

  int doit_k_induction();
  int doit_k_induction_parallel();

  int doit_falsification();
  int doit_incremental();
  int doit_termination();

  int do_base_case(
    optionst &opts,
    goto_functionst &goto_functions,
    const BigInt &k_step);

  int do_forward_condition(
    optionst &opts,
    goto_functionst &goto_functions,
    const BigInt &k_step);

  int do_inductive_step(
    optionst &opts,
    goto_functionst &goto_functions,
    const BigInt &k_step);

  bool read_goto_binary(goto_functionst &goto_functions);

  bool set_claims(goto_functionst &goto_functions);

  void set_verbosity_msg();

  uint64_t read_time_spec(const char *str);
  uint64_t read_mem_spec(const char *str);

  void preprocessing();

  void print_ileave_points(namespacet &ns, goto_functionst &goto_functions);

  FILE *out = stdout;
  FILE *err = stderr;

  std::vector<std::unique_ptr<goto_functions_algorithm>>
    goto_preprocess_algorithms;

private:
  void close_file(FILE *f)
  {
    if(f != stdout && f != stderr)
    {
      fclose(f);
    }
  }

public:
  goto_functionst goto_functions;
};

#endif
