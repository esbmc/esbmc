#include <goto-symex/witnesses.h>
#include <ac_config.h>
#include <boost/property_tree/ptree.hpp>
#include <fstream>
#include <langapi/languages.h>
#include <util/irep2.h>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <boost/version.hpp>

typedef boost::property_tree::ptree xmlnodet;

short int nodet::_id = 0;
short int edget::_id = 0;

void grapht::generate_graphml(optionst & options)
{
  xmlnodet graphml_node;
  create_graphml(graphml_node);

  xmlnodet graph_node;
  if (this->witness_type == grapht::VIOLATION)
    create_violation_graph_node(this->verified_file, options, graph_node);
  else
	create_correctness_graph_node(this->verified_file, options, graph_node);

  nodet * prev_node = nullptr;
  for (auto &current_edge : this->edges)
  {
     if (prev_node == nullptr || prev_node != current_edge.from_node)
     {
       xmlnodet from_node_node;
       create_node_node(*current_edge.from_node, from_node_node);
       graph_node.add_child("node", from_node_node);
     }
     xmlnodet to_node_node;
     create_node_node(*current_edge.to_node, to_node_node);
     graph_node.add_child("node", to_node_node);
     xmlnodet edge_node;
     create_edge_node(current_edge, edge_node);
     graph_node.add_child("edge", edge_node);
     prev_node = current_edge.to_node;
  }
  graphml_node.add_child("graphml.graph", graph_node);

#if (BOOST_VERSION >= 105700)
  boost::property_tree::xml_writer_settings<std::string> settings(' ', 2);
#else
  boost::property_tree::xml_writer_settings<char> settings(' ', 2);
#endif
  boost::property_tree::write_xml(options.get_option("witness-output"), graphml_node, std::locale(), settings);
}

/* */
void grapht::check_create_new_thread(uint16_t thread_id, nodet * prev_node)
{
  if (std::find(std::begin(this->threads), std::end(this->threads), thread_id) == std::end(this->threads))
  {
    this->threads.push_back(thread_id);
    nodet * new_node = new nodet();
    edget * new_edge = new edget();
    new_edge->create_thread = std::to_string(thread_id);
    new_edge->from_node = prev_node;
    new_edge->to_node = new_node;
    this->edges.push_back(*new_edge);
    prev_node = new_node;
  }
}
/* */
void grapht::create_initial_edge()
{
  nodet * first_node = new nodet();
  first_node->entry = true;
  nodet * initial_node = new nodet();;
  edget first_edge(first_node, initial_node);
  first_edge.enter_function = "main";
  first_edge.create_thread = std::to_string(0);
  this->threads.push_back(0);
  this->edges.push_back(first_edge);
}

/* */
int generate_sha1_hash_for_file(const char * path, std::string & output)
{
  FILE * file = fopen(path, "rb");
  if(!file)
    return -1;

  const int bufSize = 32768;
  char * buffer = (char *) alloca(bufSize);

  crypto_hash c;
  int bytesRead = 0;
  while((bytesRead = fread(buffer, 1, bufSize, file)))
    c.ingest(buffer, bytesRead);

  c.fin();
  output = c.to_string();

  fclose(file);
  return 0;
}

/* */
std::string execute_cmd(const std::string& command)
{
  /* add ./ for linux execution */
  std::string initial = command.substr(0, 1);
  FILE* pipe = popen(command.c_str(), "r");
  if (!pipe)
    return "ERROR";
  char buffer[128];
  std::string result;
  while (!feof(pipe))
  {
    if (fgets(buffer, 128, pipe) != nullptr)
      result += buffer;
  }
  pclose(pipe);
  return result;
}

/* */
std::string read_file(const std::string& path)
{
  std::ifstream t(path.c_str());
  std::string str((std::istreambuf_iterator<char>(t)),
      std::istreambuf_iterator<char>());
  return str;
}

/* */
std::string trim(const std::string& str)
{
  const std::string whitespace_characters = " \t\r\n";
  size_t first_non_whitespace = str.find_first_not_of(whitespace_characters);
  if (first_non_whitespace == std::string::npos)
    return "";
  size_t last_non_whitespace = str.find_last_not_of(whitespace_characters);
  size_t length = last_non_whitespace - first_non_whitespace + 1;
  return str.substr(first_non_whitespace, length);
}

/* */
void map_line_number_to_content(const std::string& source_code_file,
    std::map<int, std::string> & line_content_map)
{
  std::ifstream sfile(source_code_file);
  if (!sfile)
  {
    return;
  }
  std::string source_content = read_file(source_code_file);
  std::istringstream source_stream(source_content);
  std::string line;
  int line_count = 0;
  while (std::getline(source_stream, line))
  {
    line_count++;
    line_content_map[line_count] = trim(line);
  }
}

/* */
void create_node_node(
  nodet & node,
  xmlnodet & nodenode)
{
  nodenode.add("<xmlattr>.id", node.id);
  if (node.violation)
  {
    xmlnodet data_violation;
    data_violation.add("<xmlattr>.key", "violation");
    data_violation.put_value("true");
    nodenode.add_child("data", data_violation);
  }
  if (node.sink)
  {
    xmlnodet data_sink;
    data_sink.add("<xmlattr>.key", "sink");
    data_sink.put_value("true");
    nodenode.add_child("data", data_sink);
  }
  if (node.entry)
  {
    xmlnodet data_entry;
    data_entry.add("<xmlattr>.key", "entry");
    data_entry.put_value("true");
    nodenode.add_child("data", data_entry);
  }
  if (node.cycle_head)
  {
    xmlnodet data_cycle_head;
    data_cycle_head.add("<xmlattr>.key", "cyclehead");
    data_cycle_head.put_value("true");
    nodenode.add_child("data", data_cycle_head);
  }
  if (!node.invariant.empty())
  {
    xmlnodet data_invariant;
    data_invariant.add("<xmlattr>.key", "invariant");
    data_invariant.put_value(node.invariant);
    nodenode.add_child("data", data_invariant);
  }
  if (!node.invariant_scope.empty())
  {
    xmlnodet data_invariant;
    data_invariant.add("<xmlattr>.key", "invariant.scope");
    data_invariant.put_value(node.invariant_scope);
    nodenode.add_child("data", data_invariant);
  }
}

/* */
void create_edge_node(edget & edge, xmlnodet & edgenode)
{
  edgenode.add("<xmlattr>.id", edge.id);
  edgenode.add("<xmlattr>.source", edge.from_node->id);
  edgenode.add("<xmlattr>.target", edge.to_node->id);
  if (edge.start_line != c_nonset)
  {
    xmlnodet data_lineNumberInOrigin;
    data_lineNumberInOrigin.add("<xmlattr>.key", "startline");
    data_lineNumberInOrigin.put_value(edge.start_line);
    edgenode.add_child("data", data_lineNumberInOrigin);
  }
  if (edge.end_line != c_nonset)
  {
    xmlnodet data_endLine;
    data_endLine.add("<xmlattr>.key", "endline");
    data_endLine.put_value(edge.end_line);
    edgenode.add_child("data", data_endLine);
  }
  if (edge.start_offset != c_nonset)
  {
    xmlnodet data_startoffset;
    data_startoffset.add("<xmlattr>.key", "startoffset");
    data_startoffset.put_value(edge.start_offset);
    edgenode.add_child("data", data_startoffset);
  }
  if (edge.end_offset != c_nonset)
  {
    xmlnodet data_endoffset;
    data_endoffset.add("<xmlattr>.key", "endoffset");
    data_endoffset.put_value(edge.end_offset);
    edgenode.add_child("data", data_endoffset);
  }
  if (!edge.return_from_function.empty())
  {
    xmlnodet data_returnFromFunction;
    data_returnFromFunction.add("<xmlattr>.key", "returnFromFunction");
    data_returnFromFunction.put_value(edge.return_from_function);
    edgenode.add_child("data", data_returnFromFunction);
  }
  if (!edge.enter_function.empty())
  {
    xmlnodet data_enterFunction;
    data_enterFunction.add("<xmlattr>.key", "enterFunction");
    data_enterFunction.put_value(edge.enter_function);
    edgenode.add_child("data", data_enterFunction);
  }
  if (!edge.assumption.empty())
  {
    xmlnodet data_assumption;
    data_assumption.add("<xmlattr>.key", "assumption");
    data_assumption.put_value(edge.assumption);
    edgenode.add_child("data", data_assumption);
  }
  if (!edge.assumption_scope.empty())
  {
    xmlnodet data_assumptionScope;
    data_assumptionScope.add("<xmlattr>.key", "assumption.scope");
    data_assumptionScope.put_value(edge.assumption_scope);
    edgenode.add_child("data", data_assumptionScope);
  }
  if (!edge.thread_id.empty())
  {
    xmlnodet data_thread_id;
    data_thread_id.add("<xmlattr>.key", "threadId");
    data_thread_id.put_value(edge.thread_id);
    edgenode.add_child("data", data_thread_id);
  }
  if (!edge.create_thread.empty())
  {
    xmlnodet data_create_thread;
    data_create_thread.add("<xmlattr>.key", "createThread");
    data_create_thread.put_value(edge.create_thread);
    edgenode.add_child("data", data_create_thread);
  }
}

/* */
void create_graphml(xmlnodet & graphml)
{
  graphml.add("graphml.<xmlattr>.xmlns",
    "http://graphml.graphdrawing.org/xmlns");
  graphml.add("graphml.<xmlattr>.xmlns:xsi",
    "http://www.w3.org/2001/XMLSchema-instance");

  xmlnodet frontier_node;
  frontier_node.add("<xmlattr>.id", "frontier");
  frontier_node.put(
  xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "isFrontierNode");
  frontier_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "boolean");
  frontier_node.add("<xmlattr>.for", "node");
  xmlnodet frontier_default_node;
  frontier_default_node.put_value("false");
  frontier_node.add_child("default", frontier_default_node);
  graphml.add_child("graphml.key", frontier_node);

  xmlnodet violation_node;
  violation_node.add("<xmlattr>.id", "violation");
  violation_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "isViolationNode");
  violation_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "boolean");
  violation_node.add("<xmlattr>.for", "node");
  xmlnodet violation_default_node;
  violation_default_node.put_value("false");
  violation_node.add_child("default", violation_default_node);
  graphml.add_child("graphml.key", violation_node);

  xmlnodet entry_node;
  entry_node.add("<xmlattr>.id", "entry");
  entry_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "isEntryNode");
  entry_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "boolean");
  entry_node.add("<xmlattr>.for", "node");
  xmlnodet entry_default_node;
  entry_default_node.put_value("false");
  entry_node.add_child("default", entry_default_node);
  graphml.add_child("graphml.key", entry_node);

  xmlnodet sink_node;
  sink_node.add("<xmlattr>.id", "sink");
  sink_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "isSinkNode");
  sink_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "boolean");
  sink_node.add("<xmlattr>.for", "node");
  xmlnodet sink_default_node;
  sink_default_node.put_value("false");
  sink_node.add_child("default", sink_default_node);
  graphml.add_child("graphml.key", sink_node);

  xmlnodet cycle_head_node;
  cycle_head_node.add("<xmlattr>.id", "cyclehead");
  cycle_head_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "cyclehead");
  cycle_head_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "boolean");
  cycle_head_node.add("<xmlattr>.for", "node");
  xmlnodet cycle_head_default_node;
  cycle_head_default_node.put_value("false");
  cycle_head_node.add_child("default", cycle_head_default_node);
  graphml.add_child("graphml.key", cycle_head_node);

  xmlnodet source_code_lang_node;
  source_code_lang_node.add("<xmlattr>.id", "sourcecodelang");
  source_code_lang_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "sourcecodeLanguage");
  source_code_lang_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  source_code_lang_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", source_code_lang_node);

  xmlnodet program_file_node;
  program_file_node.add("<xmlattr>.id", "programfile");
  program_file_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "programfile");
  program_file_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  program_file_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", program_file_node);

  xmlnodet program_hash_node;
  program_hash_node.add("<xmlattr>.id", "programhash");
  program_hash_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "programhash");
  program_hash_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  program_hash_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", program_hash_node);

  xmlnodet creation_time_node;
  creation_time_node.add("<xmlattr>.id", "creationtime");
  creation_time_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "creationtime");
  creation_time_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  creation_time_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", creation_time_node);

  xmlnodet specification_node;
  specification_node.add("<xmlattr>.id", "specification");
  specification_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "specification");
  specification_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  specification_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", specification_node);

  xmlnodet architecture_node;
  architecture_node.add("<xmlattr>.id", "architecture");
  architecture_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "architecture");
  architecture_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  architecture_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", architecture_node);

  xmlnodet producer_node;
  producer_node.add("<xmlattr>.id", "producer");
  producer_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "producer");
  producer_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  producer_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", producer_node);

  xmlnodet source_code_node;
  source_code_node.add("<xmlattr>.id", "sourcecode");
  source_code_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "sourcecode");
  source_code_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  source_code_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", source_code_node);

  xmlnodet start_line_node;
  start_line_node.add("<xmlattr>.id", "startline");
  start_line_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "startline");
  start_line_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "int");
  start_line_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", start_line_node);

  xmlnodet start_offset_node;
  start_offset_node.add("<xmlattr>.id", "startoffset");
  start_offset_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "startoffset");
  start_offset_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "int");
  start_offset_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", start_offset_node);

  xmlnodet control_node;
  control_node.add("<xmlattr>.id", "control");
  control_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "control");
  control_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  control_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", control_node);

  xmlnodet invariant_node;
  invariant_node.add("<xmlattr>.id", "invariant");
  invariant_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "invariant");
  invariant_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  invariant_node.add("<xmlattr>.for", "node");
  graphml.add_child("graphml.key", invariant_node);

  xmlnodet invariant_scope_node;
  invariant_scope_node.add("<xmlattr>.id", "invariant.scope");
  invariant_scope_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "invariant.scope");
  invariant_scope_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  invariant_scope_node.add("<xmlattr>.for", "node");
  graphml.add_child("graphml.key", invariant_scope_node);

  xmlnodet assumption_node;
  assumption_node.add("<xmlattr>.id", "assumption");
  assumption_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "assumption");
  assumption_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  assumption_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", assumption_node);

  xmlnodet assumption_scope_node;
  assumption_scope_node.add("<xmlattr>.id", "assumption.scope");
  assumption_scope_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "assumption");
  assumption_scope_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  assumption_scope_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", assumption_scope_node);

  xmlnodet assumption_result_function_node;
  assumption_result_function_node.add("<xmlattr>.id", "assumption.resultfunction");
  assumption_result_function_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "assumption.resultfunction");
  assumption_result_function_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  assumption_result_function_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", assumption_result_function_node);

  xmlnodet enter_function_node;
  enter_function_node.add("<xmlattr>.id", "enterFunction");
  enter_function_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "enterFunction");
  enter_function_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  enter_function_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", enter_function_node);

  xmlnodet return_from_function_node;
  return_from_function_node.add("<xmlattr>.id", "returnFromFunction");
  return_from_function_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "returnFromFunction");
  return_from_function_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  return_from_function_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", return_from_function_node);

  xmlnodet end_line_node;
  end_line_node.add("<xmlattr>.id", "endline");
  end_line_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "endline");
  end_line_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "int");
  end_line_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", end_line_node);

  xmlnodet end_offset_node;
  end_offset_node.add("<xmlattr>.id", "endoffset");
  end_offset_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "endoffset");
  end_offset_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "int");
  end_offset_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", end_offset_node);

  xmlnodet thread_id_node;
  thread_id_node.add("<xmlattr>.id", "threadId");
  thread_id_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "threadId");
  thread_id_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  thread_id_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", thread_id_node);

  xmlnodet create_thread_node;
  create_thread_node.add("<xmlattr>.id", "createThread");
  create_thread_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "createThread");
  create_thread_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  create_thread_node.add("<xmlattr>.for", "edge");
  graphml.add_child("graphml.key", create_thread_node);

  xmlnodet witness_type_node;
  witness_type_node.add("<xmlattr>.id", "witness-type");
  witness_type_node.put(
    xmlnodet::path_type("<xmlattr>|attr.name", '|'),
    "witness-type");
  witness_type_node.put(
    xmlnodet::path_type("<xmlattr>|attr.type", '|'),
    "string");
  witness_type_node.add("<xmlattr>.for", "graph");
  graphml.add_child("graphml.key", witness_type_node);
}

/* */
void _create_graph_node(
  std::string & verifiedfile,
  optionst & options,
  xmlnodet & graphnode )
 {
  graphnode.add("<xmlattr>.edgedefault", "directed");

  xmlnodet pProducer;
  pProducer.add("<xmlattr>.key", "producer");

  std::string producer = options.get_option("witness-producer");
  if (producer.empty())
  {
    producer = "ESBMC " + std::string(ESBMC_VERSION);
    if(options.get_bool_option("k-induction"))
      producer += " kind";
    else if(options.get_bool_option("k-induction-parallel"))
      producer += " kind";
    else if(options.get_bool_option("falsification"))
      producer += " falsi";
    else if(options.get_bool_option("incremental-bmc"))
      producer += " incr";
  }

  pProducer.put_value(producer);

  graphnode.add_child("data", pProducer);

  xmlnodet pSourceCodeLang;
  pSourceCodeLang.add("<xmlattr>.key", "sourcecodelang");
  pSourceCodeLang.put_value("C");
  graphnode.add_child("data", pSourceCodeLang);

  xmlnodet pArchitecture;
  pArchitecture.add("<xmlattr>.key", "architecture");
  pArchitecture.put_value(std::to_string(config.ansi_c.word_size) + "bit");
  graphnode.add_child("data", pArchitecture);

  xmlnodet pProgramFile;
  pProgramFile.add("<xmlattr>.key", "programfile");
  std::string program_file = options.get_option("witness-programfile");
  if (program_file.empty())
    pProgramFile.put_value(verifiedfile);
  else
    pProgramFile.put_value(program_file);
  graphnode.add_child("data", pProgramFile);

  std::string programFileHash;
  if (program_file.empty())
    generate_sha1_hash_for_file(verifiedfile.c_str(), programFileHash);
  else
    generate_sha1_hash_for_file(program_file.c_str(), programFileHash);
  xmlnodet pProgramHash;
  pProgramHash.add("<xmlattr>.key", "programhash");
  pProgramHash.put_value(programFileHash);
  graphnode.add_child("data", pProgramHash);

  xmlnodet pDataSpecification;
  pDataSpecification.add("<xmlattr>.key", "specification");
  if (options.get_bool_option("overflow-check"))
    pDataSpecification.put_value("CHECK( init(main()), LTL(G ! overflow) )");
  else if (options.get_bool_option("memory-leak-check"))
    pDataSpecification.put_value("CHECK( init(main()), LTL(G valid-free|valid-deref|valid-memtrack) )");
  else
    pDataSpecification.put_value("CHECK( init(main()), LTL(G ! call(__VERIFIER_error())) )");
  graphnode.add_child("data", pDataSpecification);

  boost::posix_time::ptime creation_time = boost::posix_time::microsec_clock::universal_time();
  xmlnodet p_creationTime;
  p_creationTime.add("<xmlattr>.key", "creationtime");
  p_creationTime.put_value(boost::posix_time::to_iso_extended_string(creation_time));
  graphnode.add_child("data", p_creationTime);
}

/* */
void create_violation_graph_node(
  std::string & verifiedfile,
  optionst & options,
  xmlnodet & graphnode )
{
  _create_graph_node(verifiedfile, options, graphnode);
  xmlnodet pWitnessType;
  pWitnessType.add("<xmlattr>.key", "witness-type");
  pWitnessType.put_value("violation_witness");
  graphnode.add_child("data", pWitnessType);
}

/* */
void create_correctness_graph_node(
  std::string & verifiedfile,
  optionst & options,
  xmlnodet & graphnode )
{
  _create_graph_node(verifiedfile, options, graphnode);
  xmlnodet pWitnessType;
  pWitnessType.add("<xmlattr>.key", "witness-type");
  pWitnessType.put_value("correctness_witness");
  graphnode.add_child("data", pWitnessType);
}

const std::regex regex_array("[a-zA-Z0-9_]+ = \\{ ?(-?[0-9]+(.[0-9]+)?,? ?)+ ?\\};");

/* */
void reformat_assignment_array(
  const namespacet & ns,
  const goto_trace_stept & step,
  std::string & assignment)
{
  std::regex re{R"(((-?[0-9]+(.[0-9]+)?)))"};
  using reg_itr = std::regex_token_iterator<std::string::iterator>;
  uint16_t pos = 0;
  std::string lhs = from_expr(ns, "", step.lhs);
  std::string assignment_array = "";
  for (reg_itr it{assignment.begin(), assignment.end(), re, 1}, end{}; it != end;) {
    std::string value = *it++;
    assignment_array += lhs + "[" + std::to_string(pos++) + "] = " + value + "; ";
  }
  assignment_array.pop_back();
  assignment = assignment_array;
}

const std::regex regex_structs("[a-zA-Z0-9_]+ = \\{ ?(\\.([a-zA-Z0-9_]+)=(-?[0-9]+(.[0-9]+)?),? ?)+\\};");

/* */
void reformat_assignment_structs(
  const namespacet & ns,
  const goto_trace_stept & step,
  std::string & assignment)
{
  std::regex re{R"((((.([a-zA-Z0-9_]+)=(-?[0-9]+(.[0-9]+)?))+)))"};
  using reg_itr = std::regex_token_iterator<std::string::iterator>;
  std::string lhs = from_expr(ns, "", step.lhs);
  std::string assignment_struct = "";
  for (reg_itr it{assignment.begin(), assignment.end(), re, 1}, end{}; it != end;) {
   std::string a = *it++;
   assignment_struct += lhs + a + "; ";
  }
  assignment_struct.pop_back();
  assignment = assignment_struct;
}

/* */
void check_replace_invalid_assignment(std::string & assignment)
{
  /* replace: SAME-OBJECT(&var1, &var2) into &var1 == &var2 (XXX check if should stay) */
  //std::regex e ("SAME-OBJECT\\((&([a-zA-Z_0-9]+)), (&([a-zA-Z_0-9]+))\\)");
  //assignment = std::regex_replace(assignment, e ,"$1 == $3");
  std::smatch m;
  /* looking for undesired in the assignment */
  if (std::regex_search(assignment,m,std::regex("&dynamic_([0-9]+)_value")) ||
      std::regex_search(assignment,m,std::regex("dynamic_([0-9]+)_array")) ||
      std::regex_search(assignment,m,std::regex("anonymous at")) ||
      std::regex_search(assignment,m,std::regex("Union")) ||
      std::regex_search(assignment,m,std::regex("&")) ||
      std::regex_search(assignment,m,std::regex("@")) ||
      std::regex_search(assignment,m,std::regex("POINTER_OFFSET")) ||
      std::regex_search(assignment,m,std::regex("SAME-OBJECT")) ||
      std::regex_search(assignment,m,std::regex("BITCAST:")))
    assignment.clear();
}

/* */
std::string get_formated_assignment(const namespacet & ns, const goto_trace_stept & step)
{
  std::string assignment = "";
  if(!is_nil_expr(step.value) && (is_valid_witness_step(ns, step)))
  {
    assignment += from_expr(ns, "", step.lhs);
    assignment += " = ";
    assignment += from_expr(ns, "", step.value);
    assignment += ";";
    if (std::regex_match(assignment, regex_array))
      reformat_assignment_array(ns, step, assignment);
    else if (std::regex_match(assignment, regex_structs))
      reformat_assignment_structs(ns, step, assignment);
    check_replace_invalid_assignment(assignment);
  }
  return assignment;
}

/* */
std::string w_string_replace(
  std::string subject,
  const std::string & search,
  const std::string & replace)
{
  size_t pos = 0;
  while ((pos = subject.find(search, pos)) != std::string::npos) {
     subject.replace(pos, search.length(), replace);
     pos += replace.length();
  }
  return subject;
}

/* */
void get_offsets(
  const std::string & file_path,
  const uint16_t line_number,
  uint16_t & p_startoffset,
  uint16_t & p_endoffset)
{
  uint16_t startoffset = 0;
  uint16_t endoffset = 0;
  uint8_t whiteSpaces = 0;
  size_t endOfAssignment = std::string::npos;
  std::string line;
  std::ifstream file (file_path.c_str());

  if (file.is_open())
  {
    for (int currentLineNumber = 1; getline(file,line) && (endOfAssignment == std::string::npos); currentLineNumber++)
    {
      line += '\n';
      if (currentLineNumber >= line_number)
      {
        if (currentLineNumber == line_number)
        {
          endoffset = line.size();
          for(; line.at(whiteSpaces) == ' ' && (whiteSpaces < line.size()); whiteSpaces++)
            startoffset += whiteSpaces;
        }
        endOfAssignment = line.rfind(';');
      }
      else
      {
        startoffset += line.size();
      }
    }
    endoffset += startoffset;
    file.close();
  }
  p_startoffset = startoffset;
  p_endoffset = endoffset;
}

/* */
bool is_valid_witness_step(
  const namespacet &ns,
  const goto_trace_stept & step)
{
  languagest languages(ns, "C");
  std::string lhsexpr;
  languages.from_expr(migrate_expr_back(step.lhs), lhsexpr);
  std::string location = step.pc->location.to_string();
  return ((location.find("built-in") & location.find("library") &
           lhsexpr.find("__ESBMC") & lhsexpr.find("stdin") &
           lhsexpr.find("stdout") & lhsexpr.find("stderr") &
		   lhsexpr.find("$") & lhsexpr.find("sys_")) == std::string::npos);
}

/* */
bool is_valid_witness_expr(
  const namespacet &ns,
	const irep_container<expr2t> & exp)
{
  languagest languages(ns, "C");
  std::string value;
  languages.from_expr(migrate_expr_back(exp), value);
  return (value.find("__ESBMC") &
    value.find("stdin")         &
    value.find("stdout")        &
    value.find("stderr")        &
    value.find("sys_")) == std::string::npos;
}

/* */
uint16_t get_line_number(
  std::string& verified_file,
  uint16_t relative_line_number,
  optionst & options)
{
  std::string program_file = options.get_option("witness-programfile");
  /* check if it is necessary to get the relative line */
  if (program_file.empty() || verified_file == program_file)
  {
    return relative_line_number;
  }
  std::string line;
  std::string relative_content;
  std::ifstream stream_relative (verified_file);
  std::ifstream stream_programfile (program_file);
  uint16_t line_count = 0;
  /* get the relative content */
  if (stream_relative.is_open())
  {
	while(getline(stream_relative, line) &&
		  line_count < relative_line_number)
	{
	  relative_content = line;
	  line_count++;
	}
  }
  /* file for the line in the programfile */
  line_count = 1;
  if (stream_programfile.is_open())
  {
    while(getline(stream_programfile, line) &&
  	  line != relative_content)
    {
      line_count++;
    }
  }
  return line_count;
}

std::string read_line(std::string file, uint16_t line_number)
{
  std::string line;
  std::string line_code = "";
  std::ifstream stream(file);
  uint16_t line_count = 0;
  if (stream.is_open())
  {
	while(getline(stream, line) &&
		  line_count < line_number)
	{
      line_code = line;
	  line_count++;
	}
  }
  return line_code;
}

const std::regex regex_invariants("( +)?(__((VERIFIER|ESBMC))_)?(assume|assert)\\([a-zA-Z(-?(0-9))\\[\\]_>=+/*<~.&! \\(\\)]+\\);( +)?");

std::string get_invariant(
  std::string verified_file,
  uint16_t line_number,
  optionst & options )
{
  std::string invariant = "";
  std::string line_code = "";

  std::string program_file = options.get_option("witness-programfile");
  if (program_file.empty() || verified_file == program_file)
  {
    line_code = read_line(verified_file, line_number);
  }
  else
  {
    uint16_t program_file_line_number = get_line_number(verified_file, line_number, options);
    line_code = read_line(program_file, program_file_line_number);
  }
  if (std::regex_match(line_code, regex_invariants))
  {
    std::regex re("(\\([a-zA-Z(-?(0-9))\\[\\]_>=+/*<~.&! \\(\\)]+\\))");
    using reg_itr = std::regex_token_iterator<std::string::iterator>;
    for (reg_itr it{line_code.begin(), line_code.end(), re, 1}, end{}; it != end;) {
      invariant = *it++;
      break;
    }
  }
  return invariant;
}
