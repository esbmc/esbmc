#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <util/namespace.h>
#include <util/irep2.h>
#include <langapi/language_util.h>
#include <goto_trace.h>
#include <string>
#include <regex>

typedef boost::property_tree::ptree xmlnodet;

#define lightweight_witness
#define c_nonset 0xFFFF

class nodet
{
private:
  static short int _id;

public:
  std::string id;
  bool entry = false;
  bool sink = false;
  bool violation = false;
  bool cycle_head = false;
  std::string invariant;
  std::string invariant_scope;
  nodet(void)
  {
    id = "N" + std::to_string(_id);
    _id++;
  }
};

class edget
{
private:
  static short int _id;

public:
  std::string id;
  std::string assumption;
  std::string assumption_scope;
  std::string assumption_resultfunction;
  std::string enter_function;
  std::string return_from_function;
  std::string thread_id;
  std::string create_thread;
  unsigned short int start_line = c_nonset;
  unsigned short int end_line = c_nonset;
  unsigned short int start_offset = c_nonset;
  unsigned short int end_offset = c_nonset;
  bool control = false;
  bool enter_loop_head = false;
  nodet *from_node;
  nodet *to_node;
  edget(void)
  {
    id = "E" + std::to_string(_id);
    _id++;
    from_node = NULL;
    to_node = NULL;
  }
  edget(nodet *from_node, nodet *to_node)
  {
    id = "E" + std::to_string(_id);
    _id++;
    this->from_node = from_node;
    this->to_node = to_node;
  }
};

class grapht
{
private:
  std::vector<uint16_t> threads;
  void create_initial_edge();

public:
  enum typet
  {
    VIOLATION,
    CORRECTNESS
  };
  typet witness_type;
  std::string verified_file;
  std::vector<edget> edges;
  grapht(typet t)
  {
    witness_type = t;
    create_initial_edge();
  }
  void generate_graphml(optionst &options);
  void check_create_new_thread(uint16_t thread_id, nodet *prev_node);
};

/**
 * Create a GraphML node, which is the most external
 * one and includes graph, edges, and nodes.
 */
void create_graphml(xmlnodet &graphml);

/**
 * Create a violation graph node.
 *
 * This node contains all edges and vertexes
 * of the GraphML requested by SVCOMP.
 */
void create_violation_graph_node(
  std::string &verifiedfile,
  optionst &options,
  xmlnodet &graphnode);

/**
 * Create a correctness graph node.
 *
 * See create_violation_graph_node().
 */
void create_correctness_graph_node(
  std::string &verifiedfile,
  optionst &options,
  xmlnodet &graphnode);

/**
 * Create a edge node.
 *
 * This node contains information about
 * lines, offsets, assumptions, invariants, and etc.
 */
void create_edge_node(edget &edge, xmlnodet &edgenode);

/**
 * Create a node node.
 */
void create_node_node(nodet &node, xmlnodet &nodenode);

/**
 * This function checks if the current counterexample step
 * is valid for the GraphML. A priori, ESBMC only prints steps
 * from the original program (i.e., internals and built-in
 * are excluded).
 */
bool is_valid_witness_step(const namespacet &ns, const goto_trace_stept &step);

/**
 * If the current step is an assignment, this function
 * will return the lhs and rhs formated in a way expected
 * by the assumption field.
 */
std::string
get_formated_assignment(const namespacet &ns, const goto_trace_stept &step);

/**
 *
 */
std::string w_string_replace(
  std::string subject,
  const std::string &search,
  const std::string &replace);

/**
 *
 */
void get_offsets(
  const std::string &file_path,
  const uint16_t line_number,
  uint16_t &p_startoffset,
  uint16_t &p_endoffset);

/**
 *
 */
bool is_valid_witness_expr(
  const namespacet &ns,
  const irep_container<expr2t> &exp);

/**
 *
 */
uint16_t get_line_number(
  std::string &verified_file,
  uint16_t relative_line_number,
  optionst &options);

/**
 *
 */
int generate_sha1_hash_for_file(const char *path, std::string &output);

/**
 *
 */
void map_line_number_to_content(
  const std::string &source_code_file,
  std::map<int, std::string> &line_content_map);

/**
 *
 */
std::string get_invariant(
  std::string verified_file,
  uint16_t line_number,
  optionst &options);
