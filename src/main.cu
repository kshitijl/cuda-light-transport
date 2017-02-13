#include <iostream>
#include <fstream>

#include <iomanip>
#include <ctime>
#include <sstream>

#include <string>

#include <assert.h>

#include "simple-interop.hxx"
#include "write-ppm.hxx"

#include "render.hxx"

#include <GL/freeglut_ext.h>

#include <json.hpp>

using json = nlohmann::json;
int main(int argc, char **argv) {
  std::ifstream infile{argv[1]};
  std::string cc{std::istreambuf_iterator<char>(infile),
      std::istreambuf_iterator<char>()};
  auto j = json::parse(cc.c_str());

  for(auto& element : j) {
    if(element.count("emittance"))
      std::cout << element["emittance"] << " ";
    std::cout << element["center"] << "\n";
  }

  
  return 0;
}
