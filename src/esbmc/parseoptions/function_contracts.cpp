#include <ac_config.h>

#include <esbmc/esbmc_parseoptions.h>
#include <goto-programs/contracts/contracts.h>
#include <util/irep/irep.h>
#include <util/symtab/symbol.h>
#include <list>
#include <set>
#include <string>

// Process function contracts if enabled. Returns true on a usage error.
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

  // Lambda function to process function list (handles "*" wildcard).
  // An unresolvable name is a usage error rather than a warning: these options
  // decide *what* gets verified, so silently skipping one leaves ESBMC
  // verifying something other than what was asked and still reporting a
  // verdict for it. Sets ok=false on the first bad name.
  bool ok = true;
  auto process_function_list =
    [&collect_functions_with_contracts, &contracts, &ok](
      const std::list<std::string> &func_list, const char *opt) {
      std::set<std::string> result;
      for (const auto &func : func_list)
      {
        if (func == "*")
        {
          // "*" means all functions with contracts
          return collect_functions_with_contracts();
        }

        if (contracts.find_function_symbol(func) == nullptr)
        {
          log_error(
            "--{}: no function named '{}'{}",
            opt,
            func,
            func.find(',') != std::string::npos
              ? " (names are not comma-separated; repeat the flag once per "
                "function, or use \"*\")"
              : "");
          ok = false;
          return std::set<std::string>();
        }
        result.insert(func);
      }
      return result;
    };

  // Process enforce-contract option
  if (has_enforce)
  {
    const std::list<std::string> &enforce_list =
      cmdline.get_values("enforce-contract");
    std::set<std::string> to_enforce =
      process_function_list(enforce_list, "enforce-contract");
    if (!ok)
      return true;

    if (!to_enforce.empty())
    {
      log_status("Enforcing contracts for {} function(s)", to_enforce.size());
      // Pass --function entry point so the enforce wrapper allocates fresh
      // backing storage for pointer params (harness receives nil args).
      std::string entry_function =
        cmdline.isset("function") ? cmdline.getval("function") : "";
      // Assigns compliance check is always enabled: without it, functions can
      // lie about their assigns clause, causing false VERIFICATION SUCCESSFUL.
      contracts.enforce_contracts(to_enforce, entry_function, true);
    }
  }

  // Process replace-call-with-contract option
  if (has_replace)
  {
    const std::list<std::string> &replace_list =
      cmdline.get_values("replace-call-with-contract");
    std::set<std::string> to_replace =
      process_function_list(replace_list, "replace-call-with-contract");
    if (!ok)
      return true;

    if (!to_replace.empty())
      contracts.replace_calls(to_replace);
  }

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

  // Process --enforce-all-contracts
  if (has_enforce_all)
  {
    std::set<std::string> to_enforce = collect_annotated_contract_functions();
    if (!to_enforce.empty())
    {
      log_status(
        "Enforcing annotated contracts for {} function(s)", to_enforce.size());
      std::string entry_function =
        cmdline.isset("function") ? cmdline.getval("function") : "";
      contracts.enforce_contracts(to_enforce, entry_function, true);
    }
  }

  // Process --replace-all-contracts
  if (has_replace_all)
  {
    std::set<std::string> to_replace = collect_annotated_contract_functions();
    if (!to_replace.empty())
    {
      log_status(
        "Replacing annotated calls for {} function(s)", to_replace.size());
      contracts.replace_calls(to_replace);
    }
  }

  return false;
}
