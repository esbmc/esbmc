#include <ac_config.h>

#include <esbmc/esbmc_parseoptions.h>
#include <goto-programs/contracts/contracts.h>
#include <util/irep/irep.h>
#include <util/symtab/symbol.h>
#include <list>
#include <set>
#include <string>

// Process function contracts if enabled
bool esbmc_parseoptionst::process_function_contracts(
  goto_functionst &goto_functions,
  bool has_replace,
  bool has_enforce,
  bool has_enforce_all,
  bool has_replace_all)
{
  namespacet ns(context);
  code_contractst contracts(goto_functions, context, ns);

  // Reference to context for use in lambda
  contextt &ctx = context;

  // Lambda function to collect all functions with contracts
  // This includes functions with:
  // 1. Explicit contract clauses (__ESBMC_requires, __ESBMC_ensures, __ESBMC_assigns)
  // 2. __attribute__((annotate("__ESBMC_contract"))) annotation
  auto collect_functions_with_contracts =
    [&contracts, &goto_functions, &ctx]() {
      std::set<std::string> result;
      forall_goto_functions (it, goto_functions)
      {
        if (!it->second.body_available)
          continue;

        std::string func_name = id2string(it->first);

        // Use is_compiler_generated (which correctly handles C++ USR IDs like
        // "c:@F@fst#*1I#") instead of a raw '#' string filter, which would
        // incorrectly skip all C++ functions with parameters.
        if (contracts.is_compiler_generated(func_name))
          continue;

        // Check for explicit contract clauses in function body
        if (contracts.has_contracts(it->second.body))
        {
          result.insert(func_name);
          continue;
        }

        // Check for __attribute__((annotate("__ESBMC_contract"))) annotation
        symbolt *func_sym = ctx.find_symbol(it->first);
        if (func_sym && contracts.is_annotated_contract_function(*func_sym))
        {
          result.insert(func_name);
        }
      }
      return result;
    };

  // Lambda to collect ONLY functions with __ESBMC_contract annotation
  auto collect_annotated_contract_functions =
    [&contracts, &goto_functions, &ctx]() {
      std::set<std::string> result;
      forall_goto_functions (it, goto_functions)
      {
        if (!it->second.body_available)
          continue;
        std::string func_name = id2string(it->first);
        if (contracts.is_compiler_generated(func_name))
          continue;
        symbolt *func_sym = ctx.find_symbol(it->first);
        if (func_sym && contracts.is_annotated_contract_function(*func_sym))
          result.insert(func_name);
      }
      return result;
    };

  // Lambda function to process function list (handles "*" wildcard).
  // \p named receives the names the user spelled out; "*" leaves it empty and
  // discards any name spelled alongside it. Those names cannot be checked
  // individually once "*" is in play: the wildcard resolves to full symbol IDs
  // while the user spells short names, so comparing the two would reject every
  // legitimate spelling. A misspelling next to "*" is therefore not diagnosed.
  auto process_function_list = [&collect_functions_with_contracts](
                                 const std::list<std::string> &func_list,
                                 std::set<std::string> &named) {
    std::set<std::string> result;
    for (const auto &func : func_list)
    {
      if (func == "*")
      {
        // "*" means all functions with contracts
        named.clear();
        return collect_functions_with_contracts();
      }
      result.insert(func);
      named.insert(func);
    }
    return result;
  };

  // Reject a name no pass can act on, and treat selecting nothing at all as an
  // error too. The reason comes from code_contractst::diagnose_contract_target,
  // which is where both passes' eligibility rules live, so this cannot drift
  // from what they will do -- and it is asked per option, because the two rules
  // genuinely differ (a C++ id with parameters satisfies enforce and not
  // replace).
  auto check = [&contracts](
                 const char *opt,
                 const std::set<std::string> &named,
                 const std::set<std::string> &resolved,
                 bool for_replace,
                 const char *nothing_selected) {
    if (named.empty())
    {
      if (!resolved.empty())
        return false;
      log_error("--{}: {}", opt, nothing_selected);
      return true;
    }

    bool failed = false;
    for (const auto &name : named)
    {
      std::string reason =
        contracts.diagnose_contract_target(name, for_replace);
      if (reason.empty())
        continue;
      log_error(
        "--{}: cannot use '{}': {}{}",
        opt,
        name,
        reason,
        name.find(',') != std::string::npos
          ? " (names are not comma-separated; repeat the flag once per "
            "function, or use \"*\")"
          : "");
      failed = true;
    }
    return failed;
  };

  // Resolve every requested list first and validate all of them before any
  // pass runs. enforce_contracts rewrites the functions it acts on into
  // wrappers that carry no contract, so a check made afterwards would call a
  // name that was used unusable, and would reject
  // `--enforce-contract f --replace-call-with-contract "*"`, which is how four
  // existing tests enforce one function and abstract the rest.
  std::set<std::string> enforce_named, replace_named;
  std::set<std::string> to_enforce, to_replace, annotated;

  if (has_enforce)
    to_enforce = process_function_list(
      cmdline.get_values("enforce-contract"), enforce_named);
  if (has_replace)
    to_replace = process_function_list(
      cmdline.get_values("replace-call-with-contract"), replace_named);
  if (has_enforce_all || has_replace_all)
    annotated = collect_annotated_contract_functions();

  static const char *const no_contract =
    "no function with a contract to act on";
  static const char *const no_annotation =
    "no function carries the __ESBMC_contract annotation";

  bool failed = false;
  if (has_enforce)
    failed |=
      check("enforce-contract", enforce_named, to_enforce, false, no_contract);
  if (has_replace)
    failed |= check(
      "replace-call-with-contract",
      replace_named,
      to_replace,
      true,
      no_contract);
  if (has_enforce_all)
    failed |=
      check("enforce-all-contracts", {}, annotated, false, no_annotation);
  if (has_replace_all)
    failed |=
      check("replace-all-contracts", {}, annotated, true, no_annotation);

  if (failed)
    return true;

  // Pass --function entry point so the enforce wrapper allocates fresh backing
  // storage for pointer params (harness receives nil args).
  const std::string entry_function =
    cmdline.isset("function") ? cmdline.getval("function") : "";

  // Assigns compliance check is always enabled: without it, functions can lie
  // about their assigns clause, causing false VERIFICATION SUCCESSFUL.
  if (has_enforce)
    log_status(
      "Enforcing contracts for {} function(s)",
      contracts.enforce_contracts(to_enforce, entry_function, true).size());

  if (has_replace)
    contracts.replace_calls(to_replace);

  if (has_enforce_all)
    log_status(
      "Enforcing annotated contracts for {} function(s)",
      contracts.enforce_contracts(annotated, entry_function, true).size());

  if (has_replace_all)
    contracts.replace_calls(annotated);

  return false;
}
