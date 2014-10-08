#include<iostream>
#include <boost/graph/directed_graph.hpp>
#include <boost/graph/graphml.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <string>

typedef struct graph_props {
   std::string sourcecodeLanguage;
} graph_p;

typedef struct node_props {
   std::string nodeType;
   bool isFrontierNode;
   bool isViolationNode;
   bool isEntryNode;
   bool isSinkNode;
} node_p; // = {"", 0, 0, 0 ,0};

typedef struct edge_props { 
   std::string assumption;
   std::string sourcecode; 
   std::string tokenSet;
   std::string originTokenSet;
   std::string negativeCase;
   int lineNumberInOrigin;
   std::string originFileName;
   std::string enterFunction;
   std::string returnFromFunction;
} edge_p;

int main(int, char *[]) {

  /* typedef boost::adjacency_list <boost::listS, boost::vecS, boost::directedS, node_p, edge_p, graph_p> Graph;
   typedef boost::graph_traits<Graph>::vertex_descriptor node_t;
   typedef boost::graph_traits<Graph>::edge_descriptor edge_t;
     
   Graph g;

   boost::dynamic_properties dp;
   
   dp.property("nodeType", boost::get(&node_p::nodeType, g));
   dp.property("isFrontierNode", boost::get(&node_p::isFrontierNode, g));
   dp.property("isViolationNode", boost::get(&node_p::isViolationNode, g));
   dp.property("isEntryNode", boost::get(&node_p::isEntryNode, g));
   dp.property("isSinkNode", boost::get(&node_p::isSinkNode, g)); 
  
   dp.property("assumption", boost::get(&edge_p::assumption, g));
   dp.property("sourcecode", boost::get(&edge_p::sourcecode, g));
   dp.property("tokenSet", boost::get(&edge_p::tokenSet, g));
   dp.property("originTokenSet", boost::get(&edge_p::originTokenSet, g));
   dp.property("negativeCase", boost::get(&edge_p::negativeCase, g));
   dp.property("lineNumberInOrigin", boost::get(&edge_p::lineNumberInOrigin, g));
   dp.property("originFileName", boost::get(&edge_p::enterFunction, g));
   dp.property("returnFromFunction", boost::get(&edge_p::returnFromFunction, g));
  
   node_t u = boost::add_vertex(g);
   node_t v = boost::add_vertex(g);

   edge_t e; bool b;
   boost::tie(e,b) = boost::add_edge(u,v,g);

   std::ofstream graphmlOutFile("graph.xml");
   boost::write_graphml(graphmlOutFile, g, dp, false);
   graphmlOutFile.close();
 */
   /* adjust xml */

  boost::property_tree::ptree pt;
  boost::property_tree::read_xml("graph.xml", pt, boost::property_tree::xml_parser::trim_whitespace);
  
  BOOST_FOREACH( boost::property_tree::ptree::value_type & element, pt.get_child("graphml.graph") ) {
     if( element.first == "node" ) {
	std::cout << "ACHEI UM NODE" << std::endl;

        boost::property_tree::ptree & subtree = element.second;
	boost::property_tree::ptree::const_iterator end = subtree.end();

	for (boost::property_tree::ptree::const_iterator it = subtree.begin(); it != end; ++it){
	     if( it->first == "data" ) {
  	     std::cout << it->first << " ~ DATA: " << it->second.data() << std::endl;
 		
		if ((it->second.data() == "") || (it->second.data() == "0") || (it->second.data() == "-1")){
			std::cout << "DADO VAZIO" << std::endl;
			subtree.erase("2");
		}

	     }
	}
	
	break;
     }
     if( element.first == "edge" ) {
	std::cout << "ACHEI UM EDGE" << std::endl;	
     }
  }

  boost::property_tree::xml_writer_settings<char> settings('\t', 1);;
  boost::property_tree::write_xml("graph2.xml", pt, std::locale(), settings);

}
