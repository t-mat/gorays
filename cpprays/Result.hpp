#ifndef RESULT_HPP
#define RESULT_HPP

struct Result {
  Result(size_t times)
    : samples(times, 0.0)
  {}

  std::string toJson() const {
    std::ostringstream o; // don't use auto for GCC 4.6
    o << "{\"average\":" << average() << ",";
    o << "\"samples\":[";
    for(size_t i = 0; i < samples.size(); ++i) {
      if(0 != i) {
        o << ",";
      }
      o << samples[i];
    }
    o << "]}\n";
    return o.str();
  }

  double average() const {
    return std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
  }

  std::vector<double> samples;
};

#endif
