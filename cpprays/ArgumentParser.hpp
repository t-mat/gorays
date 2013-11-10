#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP

struct ArgumentParser {
  ArgumentParser(int argc, char* argv[], std::ostream& outlog)
    : megaPixel { 1.0 }
    , times { 1 }
    , procs { getMaxThreads() }
    , outputFilename { "render.ppm" }
    , resultFilename { "result.json" }
    , artFilename { "ART" }
    , home { getEnv("RAYS_HOME") }
  {
    typedef const std::string& Arg;
    typedef std::function<void(Arg)> ArgFunc;
    typedef std::map<std::string, ArgFunc> ArgFuncMap;

    const ArgFuncMap optionMap {
        { "-mp"  , [this](Arg v) { megaPixel = std::stof(v); } }
      , { "-t"   , [this](Arg v) { times = std::stoi(v); } }
      , { "-p"   , [this](Arg v) { procs = std::stoi(v); } }
      , { "-o"   , [this](Arg v) { outputFilename = v; } }
      , { "-r"   , [this](Arg v) { resultFilename = v; } }
      , { "-a"   , [this](Arg v) { artFilename = v; } }
      , { "-home", [this](Arg v) { home = v; } }
    };
    // TODO Add usage

    const auto args = std::vector<std::string>(argv+1, argv+argc);

    const auto delim = '=';
    for(const auto& arg : args) {
      const auto pos = arg.find(delim);
      if(pos != std::string::npos) {
        const auto a = arg.substr(0, pos);
        const auto v = arg.substr(pos + 1);
        const auto it = optionMap.find(a);
        if(it != optionMap.end() && !v.empty()) {
          it->second(v);
        }
      }
    }

    if(artFilename == "ART" && !home.empty()) {
      artFilename = home + "/" + artFilename;
    }

    outputFile = std::ofstream { outputFilename };
    resultFile = std::ofstream { resultFilename };
    artFile    = std::ifstream { artFilename    };

    if (artFile.fail()) { // FIXME Use exception
      outlog << "Failed to open ART file (" << artFilename << ")" << std::endl;
      std::exit(1);
    }
  }

  static std::string usage() {
    return
      "-mp=X      [        1.0] Megapixels of the rendered image\n"
      "-t=N       [          1] Times to repeat the benchmark\n"
      "-p=N       [   #Threads] Number of render threads\n"
      "-o=FILE    [render.ppm ] Output file to write the rendered image to\n"
      "-r=FILE    [result.json] Result file to write the benchmark data to\n"
      "-a=FILE    [ART        ] the art file to use for rendering\n"
      "-home=PATH [$RAYS_HOME ] RAYS folder\n";
  }

  static std::string getEnv(const std::string& env) {
    const auto* s = std::getenv(env.c_str());
    return std::string { s ? s : "" };
  }

  static int getMaxThreads(const int defaultMaxThreads = 8) {
    const auto x = std::thread::hardware_concurrency();
    return x ? x : defaultMaxThreads;
  }

  // Don't use non-static data member initializer for GCC 4.6
  double megaPixel;
  int times;
  int procs;
  std::string outputFilename;
  std::string resultFilename;
  std::string artFilename;
  std::string home;

  mutable std::ofstream outputFile;
  mutable std::ofstream resultFile;
  mutable std::ifstream artFile;
};

#endif
