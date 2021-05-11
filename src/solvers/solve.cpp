#include <solve.h>
#include <solvers/array_conv.h>
#include <solvers/fp_conv.h>
#include <solvers/smt_array.h>
#include <solvers/smt_tuple_node.h>
#include <solvers/smt_tuple_sym.h>

const std::string list_of_all_solvers[] =
  {"z3", "boolector", "mathsat", "cvc", "yices"};

const unsigned int total_num_of_solvers =
  sizeof(list_of_all_solvers) / sizeof(std::string);

static smt_convt *pick_solver(
  bool int_encoding,
  const namespacet &ns,
  const optionst &options,
  tuple_iface **tuple_api,
  array_iface **array_api,
  fp_convt **fp_api)
{
  std::string the_solver;
  for(unsigned int i = 0; i < total_num_of_solvers; i++)
  {
    if(options.get_bool_option(list_of_all_solvers[i]))
    {
      if(the_solver != "")
      {
        std::cerr << "Please only specify one solver" << std::endl;
        abort();
      }

      the_solver = list_of_all_solvers[i];
    }
  }

  if(the_solver == "")
    the_solver = "boolector";

  return nullptr;
}

smt_convt *create_solver_factory(const namespacet &ns, const optionst &options)
{
  bool int_encoding = options.get_bool_option("int-encoding");
  tuple_iface *tuple_api = nullptr;
  array_iface *array_api = nullptr;
  fp_convt *fp_api = nullptr;
  smt_convt *ctx =
    pick_solver(int_encoding, ns, options, &tuple_api, &array_api, &fp_api);
  assert(ctx != nullptr);

  bool node_flat = options.get_bool_option("tuple-node-flattener");
  bool sym_flat = options.get_bool_option("tuple-sym-flattener");
  bool array_flat = options.get_bool_option("array-flattener");
  bool fp_to_bv = options.get_bool_option("fp2bv");

  // Pick a tuple flattener to use. If the solver has native support, and no
  // options were given, use that by default
  if(tuple_api != nullptr && !node_flat && !sym_flat)
    ctx->set_tuple_iface(tuple_api);
  // Use the node flattener if specified
  else if(node_flat)
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns));
  // Use the symbol flattener if specified
  else if(sym_flat)
    ctx->set_tuple_iface(new smt_tuple_sym_flattener(ctx, ns));
  // Default: node flattener
  else
    ctx->set_tuple_iface(new smt_tuple_node_flattener(ctx, ns));

  // Pick an array flattener to use. Again, pick the solver native one by
  // default, or the one specified, or if none of the above then use the built
  // in arrays -> to BV flattener.
  if(array_api != nullptr && !array_flat)
    ctx->set_array_iface(array_api);
  else if(array_flat)
    ctx->set_array_iface(new array_convt(ctx));
  else
    ctx->set_array_iface(new array_convt(ctx));

  if(fp_api == nullptr || fp_to_bv)
    ctx->set_fp_conv(new fp_convt(ctx));
  else
    ctx->set_fp_conv(fp_api);

  ctx->smt_post_init();
  return ctx;
}
