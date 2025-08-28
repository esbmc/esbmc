#include <ac_config.h>
#include <boost/filesystem.hpp>
#include <c2goto/cprover_library.h>
#include <cstdlib>
#include <fstream>
#include <goto-programs/goto_binary_reader.h>
#include <goto-programs/goto_functions.h>
#include <util/c_link.h>
#include <util/config.h>
#include <util/language.h>

extern "C"
{
  extern const uint8_t clib32_buf[];
  extern const uint8_t clib64_buf[];
  extern const unsigned int clib32_buf_size;
  extern const unsigned int clib64_buf_size;

  extern const uint8_t clib32_fp_buf[];
  extern const uint8_t clib64_fp_buf[];
  extern const unsigned int clib32_fp_buf_size;
  extern const unsigned int clib64_fp_buf_size;

  extern const uint8_t clib32_cherih_buf[];
  extern const uint8_t clib64_cherih_buf[];
  extern const unsigned int clib32_cherih_buf_size;
  extern const unsigned int clib64_cherih_buf_size;

  extern const uint8_t clib32_fp_cherih_buf[];
  extern const uint8_t clib64_fp_cherih_buf[];
  extern const unsigned int clib32_fp_cherih_buf_size;
  extern const unsigned int clib64_fp_cherih_buf_size;

  extern const uint8_t clib32_cherip_buf[];
  extern const uint8_t clib64_cherip_buf[];
  extern const unsigned int clib32_cherip_buf_size;
  extern const unsigned int clib64_cherip_buf_size;

  extern const uint8_t clib32_fp_cherip_buf[];
  extern const uint8_t clib64_fp_cherip_buf[];
  extern const unsigned int clib32_fp_cherip_buf_size;
  extern const unsigned int clib64_fp_cherip_buf_size;
}

namespace
{
/* [cheri][floatbv ? 1 : 0][wordsz == 64 ? 1 : 0] */
static const struct buffer
{
  const uint8_t *start;
  size_t size;
} clibs[3][2][2] = {
#ifdef ESBMC_BUNDLE_LIBC
  {
    {
      {&clib32_buf[0], clib32_buf_size},
      {&clib64_buf[0], clib64_buf_size},
    },
    {
      {&clib32_fp_buf[0], clib32_fp_buf_size},
      {&clib64_fp_buf[0], clib64_fp_buf_size},
    },
  },
  {
#  ifdef ESBMC_CHERI_HYBRID_SYSROOT
    {
      {NULL, 0}, // {&clib32_cherih_buf[0], clib32_cherih_buf_size},
      {&clib64_cherih_buf[0], clib64_cherih_buf_size},
    },
    {
      {NULL, 0}, // {&clib32_fp_cherih_buf[0], clib32_fp_cherih_buf_size},
      {&clib64_fp_cherih_buf[0], clib64_fp_cherih_buf_size},
    },
#  endif
  },
  {
#  ifdef ESBMC_CHERI_PURECAP_SYSROOT
    {
      {NULL, 0}, // {&clib32_cherip_buf[0], clib32_cherip_buf_size},
      {&clib64_cherip_buf[0], clib64_cherip_buf_size},
    },
    {
      {NULL, 0}, // {&clib32_fp_cherip_buf[0], clib32_fp_cherip_buf_size},
      {&clib64_fp_cherip_buf[0], clib64_fp_cherip_buf_size},
    },
#  endif
  },
#endif
};

// The goto reader will only pick up symbols for these functions and their dependencies
// This is a Python-specific whitelist invoked when you use set_functions_to_read
const static std::vector<std::string> python_c_models = {
  "list_init",
  "list_in_bounds",
  "list_at",
  "list_cat",
  "list_get_as",
  "list_push",
  "list_replace",
  "list_pop",
  "list_hash_string",
  "strncmp",
  "strcmp",
  "strlen",
  "ceil",
  "__ceil_array",
  "fegetround",
  "fesetround",
  "rint",
  "fesetround",
  "floor",
  "fabs",
  "sin",
  "cos",
  "exp",
  "expm1",
  "expm1_taylor",
  "fmod",
  "sqrt",
  "fmin",
  "fmax",
  "trunc",
  "frexp",
  "round",
  "copysign",
  "arctan",
  "atan",
  "_atan",
  "atan2",
  "acos",
  "arccos",
  "dot",
  "add",
  "subtract",
  "multiply",
  "divide",
  "transpose",
  "det",
  "matmul",
  "pow",
  "log",
  "pow_by_squaring",
  "log2",
  "ldexp",
  "log1p_taylor"/*,
  "__list_append__"*/};

} // namespace

/*static*/ void generate_symbol_deps(
  irep_idt name,
  irept irep,
  std::multimap<irep_idt, irep_idt> &deps)
{
  std::pair<irep_idt, irep_idt> type;

  if (irep.id() == "symbol")
  {
    type = std::pair<irep_idt, irep_idt>(name, irep.identifier());
    deps.insert(type);

    /* Cannot return here just yet
     * Symbol identifier may point to variable identifier
     * Further traversal needed to find type identifier if exists
     */
  }

  forall_irep (irep_it, irep.get_sub())
  {
    if (irep_it->id() == "argument")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->cmt_identifier());
      deps.insert(type);
    }
    else
    {
      /* Even if symbol & identifier found, further traversal might be needed for type identifier
       * Continue traversing to find symbol dependencies
       * The subcall will add the symbol identifier before traversing named and unnamed ireps so does not need to be done explicitly here
       */
      generate_symbol_deps(name, *irep_it, deps);
    }
  }

  /* The case where symbol identifier is reached but there are more nested type symbols
   * has only been seen so far when these higher-level symbols are unnamed ireps.
   *        (in particular inside an "operands" named_irep the layer above that)
   * Therefore named_irep iterator should be able to terminate on named_irep symbols
   * If there are future symbol resolution issues, consider changing this to also keep traversing
   *        For debugging, you can look at the nested structure via irept::pretty()
   */
  forall_named_irep (irep_it, irep.get_named_sub())
  {
    if (irep_it->second.id() == "symbol")
    {
      type = std::pair<irep_idt, irep_idt>(name, irep_it->second.identifier());
      deps.insert(type);
    }
    else if (irep_it->second.id() == "argument")
    {
      type =
        std::pair<irep_idt, irep_idt>(name, irep_it->second.cmt_identifier());
      deps.insert(type);
    }
    else
    {
      generate_symbol_deps(name, irep_it->second, deps);
    }
  }
}

static void ingest_symbol(
  irep_idt name,
  std::multimap<irep_idt, irep_idt> &deps,
  std::list<irep_idt> &to_include)
{
  std::pair<
    std::multimap<irep_idt, irep_idt>::const_iterator,
    std::multimap<irep_idt, irep_idt>::const_iterator>
    range;
  std::multimap<irep_idt, irep_idt>::const_iterator it;

  range = deps.equal_range(name);
  if (range.first == range.second)
    return;

  for (it = range.first; it != range.second; it++)
    to_include.push_back(it->second);

  deps.erase(name);
}

void add_cprover_library(contextt &context, const languaget *language)
{
  if (config.ansi_c.lib == configt::ansi_ct::libt::LIB_NONE)
    return;

  contextt new_ctx, store_ctx;
  goto_functionst goto_functions;
  std::multimap<irep_idt, irep_idt> symbol_deps;
  std::list<irep_idt> to_include;
  const buffer *clib;

  switch (config.ansi_c.word_size)
  {
  case 16:
    log_warning(
      "this version of ESBMC does not have a C library for 16 bit machines");
    return;
  case 32:
  case 64:
    break;
  default:
    log_error("No C library for bitwidth {}", config.ansi_c.word_size);
    abort();
  }

  clib = &clibs[config.ansi_c.cheri][!config.ansi_c.use_fixed_for_float]
               [config.ansi_c.word_size == 64];

  if (clib->size == 0)
  {
    if (language)
      return add_bundled_library_sources(context, *language);
    log_error("Zero-lengthed internal C library");
    abort();
  }

  goto_binary_reader goto_reader;

  if (language && language->id() == "python")
    goto_reader.set_functions_to_read(python_c_models);

  /* Python: actively has a function filter
   *    - not everything makes it into new_ctx
   *    - ignored symbols go into ignored_ctx
   * Other languages: no function filter
   *    - everything makes it into new_ctx
   *    - ignored_ctx empty
   */
  contextt ignored_ctx;
  if (goto_reader.read_goto_binary_array(
        clib->start, clib->size, new_ctx, ignored_ctx, goto_functions))
    abort();

  // Traverse symbols and get dependencies from both their nested types and values
  new_ctx.foreach_operand([&symbol_deps](const symbolt &s) {
    generate_symbol_deps(s.id, s.value, symbol_deps);
    generate_symbol_deps(s.id, s.type, symbol_deps);
  });

  // Add two hacks; we might use either pthread_mutex_lock or the checked
  // variant; so if one version is used, pull in the other too.
  std::pair<irep_idt, irep_idt> lockcheck(
    dstring("pthread_mutex_lock"), dstring("pthread_mutex_lock_check"));
  symbol_deps.insert(lockcheck);

  std::pair<irep_idt, irep_idt> condcheck(
    dstring("pthread_cond_wait"), dstring("pthread_cond_wait_check"));
  symbol_deps.insert(condcheck);

  std::pair<irep_idt, irep_idt> joincheck(
    dstring("pthread_join"), dstring("pthread_join_noswitch"));
  symbol_deps.insert(joincheck);

  /* Iterate through the new_ctx symbols, figure out which ones to go into store_ctx
   *    For Python this is everything: new_ctx already has a filtering layer
   *    For other frontends, only add symbols that exist already in context but value empty
   * store_ctx is what actually gets merged into the existing, final context
   */

  new_ctx.foreach_operand([&context,
                           &store_ctx,
                           &symbol_deps,
                           &to_include,
                           &language](const symbolt &s) {
    const symbolt *symbol = context.find_symbol(s.id);
    if (
      (language && language->id() == "python") ||
      (symbol != nullptr && symbol->value.is_nil()))
    {
      store_ctx.add(s);

      // ingest_symbol takes this added symbol and goes through symbol_deps
      // it only moves dependencies from symbol_deps to to_include
      //    if they're dependencies for a symbol that is definitely being included
      //    (i.e. in store_ctx)
      ingest_symbol(s.id, symbol_deps, to_include);
    }
  });

  /* Now iterate through the dependencies that we know we want to add (due to ingest_symbol filter)
   * These will be symbols that didn't make it into store_ctx
   * 
   * For Python:
   *    - symbols that didn't make it into store_ctx didn't make it because they're not in new_ctx
   *    - they will be found in ignored_ctx
   * 
   * For other frontends:
   *    - every symbol made it into new_ctx (no ignored_ctx)
   *    - not every symbol made it into store_ctx from new_ctx
   *    - they will be found in new_ctx
   */
  for (std::list<irep_idt>::const_iterator nameit = to_include.begin();
       nameit != to_include.end();
       nameit++)
  {
    symbolt *s;

    // Look in the appropriate place for this symbol
    if ((language && language->id() == "python"))
    {
      s = ignored_ctx.find_symbol(*nameit);
    }
    else
    {
      s = new_ctx.find_symbol(*nameit);
    }

    if (s != nullptr)
    {
      store_ctx.add(*s);

      /* Python frontend hasn't looked for dependencies for symbols that aren't in
       * the function whitelist, (since they're not put in new_ctx); other frontends
       * have these dependencies already available in symbol_deps.
       * Therefore add dependencies that result from this new symbol
       */
      if (language && language->id() == "python")
      {
        generate_symbol_deps(s->id, s->value, symbol_deps);
        generate_symbol_deps(s->id, s->type, symbol_deps);
      }

      ingest_symbol(*nameit, symbol_deps, to_include);
    }
  }

  // Bring store_ctx symbols into context
  if (c_link(context, store_ctx, "<built-in-library>"))
  {
    // Merging failed
    log_error("Failed to merge C library");
    abort();
  }
}
