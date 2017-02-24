/*****************************************
 * GraphML Generation
 *****************************************
 *
 *  Modifications:
 *
 *  23/01/2016 - Updated for svcomp16 according to
 *  http://sv-comp.sosy-lab.org/2016/witnesses/s3_cln1_false.witness.cpachecker.graphml
 *
 */

#include <string>

typedef struct graph_props
{
  std::string sourcecodeLanguage;
  std::string witnessType;
} graph_p;

typedef struct node_props
{
  std::string nodeType = "";
  bool isFrontierNode = false;
  bool isViolationNode = false;
  bool isEntryNode = false;
  bool isSinkNode = false;
  std::string invariant = "";
  std::string invariantScope = "";
} node_p;

typedef struct edge_props
{
  std::string assumption = "";
  std::string assumptionScope = "";
  std::string assumptionResultFunction = "";
  int startline = -1;
  int endline = -1;
  int startoffset = -1;
  int endoffset = -1;
  std::string originFileName = "";
  std::string enterFunction = "";
  std::string returnFromFunction = "";
} edge_p;

int generate_sha1_hash_for_file(const char * path, std::string & output);

void map_line_number_to_content(
  std::string source_code_file,
  std::map<int, std::string> & line_content_map);

void create_node(boost::property_tree::ptree & node, node_p & node_props);

void create_edge(
  boost::property_tree::ptree & edge,
  edge_p & edge_props,
  boost::property_tree::ptree & source,
  boost::property_tree::ptree & target);

void create_graphml(
  boost::property_tree::ptree & graphml,
  std::string file_path);

void create_graph(
  boost::property_tree::ptree & graph,
  std::string & filename,
  int & specification,
  const bool is_correctness);

std::string w_string_replace(
  std::string subject,
  const std::string & search,
  const std::string & replace);

void get_offsets_for_line_using_wc(
  const std::string & file_path,
  const int line_number,
  int & p_startoffset,
  int & p_endoffset);

bool is_valid_witness_expr(
  const namespacet &ns,
	const irep_container<expr2t> & exp);

void get_relative_line_in_programfile(
  const std::string relative_file_path,
  const int relative_line_number,
  const std::string program_file_path,
  int & programfile_line_number);
