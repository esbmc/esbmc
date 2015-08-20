#include <iostream>
#include <cstdlib>
#include <string>
#include <stdio.h>
#include <fstream>
#include <streambuf>
#include <sstream>
#include <vector>
#include <map>

std::string executable_path="../files/tokenizer";
std::string source_code="example0.c";
const std::string path = executable_path + " " + source_code;
std::string temporary_file = "/tmp/esbmc_graphml.tmp";

std::string execute_cmd(std::string command){
    FILE* pipe = popen(command.c_str(), "r");
    if (!pipe) return "ERROR";
    char buffer[128];
    std::string result = "";
    while(!feof(pipe)) {
    	if(fgets(buffer, 128, pipe) != NULL)
    		result += buffer;
    }
    pclose(pipe);
    return result;
}

std::string call_tokenize(std::string file){
   return execute_cmd(executable_path + " " + file);
}

std::string read_file( std::string path ){
   std::ifstream t(path.c_str());
   std::string str((std::istreambuf_iterator<char>(t)),std::istreambuf_iterator<char>());
   return str;
}

void write_file( std::string path, std::string content ){
   std::ofstream out(path.c_str());
   out << content;
   out.close();
}

void generate_tokens(std::string tokenized_line, std::map<int, std::string> & tokens, int & token_index){
   std::istringstream tl_stream(tokenized_line.c_str());
   std::string line;
   while (std::getline(tl_stream, line)){
      if (line != "\n" && line != ""){
         tokens[token_index] = line;
         token_index++;
      }
   }
}

int main(){

   std::string source_content = read_file(source_code);
   std::istringstream source_stream(source_content.c_str());

   std::map<int, std::map<int,std::string> > mapped_tokens;

   std::string line;
   int line_count = 0;
   int token_index = 1;
   while (std::getline(source_stream, line)){   
      line_count++;
      write_file(temporary_file, line + "\n");
      std::string tokenized_line = call_tokenize(temporary_file);

      std::map<int, std::string> tokens;
      generate_tokens(tokenized_line, tokens, token_index);
      mapped_tokens[line_count] = tokens;
   }

   std::map<int,std::string>::iterator it;
   std::map<int, std::string> tokens = mapped_tokens[4];
   for (it=tokens.begin(); it!=tokens.end(); ++it){  
      std::cout << it->first << " - " << it->second << std::endl;
   }
}
