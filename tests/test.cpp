

// DO NOT CHANGE THIS FILE, THIS IS FOR OUR TESTING PURPOSES ONLY
// But We include a sample test here for you to see how the we do the final
// testing We use functions in API to implement our tests
#include "test_api.h"
#include <iostream>
#include <vector>

// We relax the floating point comparison in the tests, since different
// implementations may have slightly different results due to precision issues.
const float EPSILON = 1e-2;

bool test_1() {
  TestAPI api;

  string input = "Hello world";
  vector<int> token_ids = api.tokenize(input);
  vector<int> expected = {128000, 9906, 1917};

  if (token_ids.size() != expected.size()) {
    std::cout << "Test failed: size mismatch. Expected " << expected.size()
              << " but got " << token_ids.size() << "\n";
    return false;
  }

  for (int i = 0; i < (int)token_ids.size(); i++) {
    if (token_ids[i] != expected[i]) {
      std::cout << "Test failed: element mismatch. Expected " << expected[i]
                << " but got " << token_ids[i] << "\n";
      return false;
    }
  }
  return true;
}

int main(int argc, char *argv[]) {
  // ANSI color codes
  const char *GREEN = "\033[32m";
  const char *RED = "\033[31m";
  const char *RESET = "\033[0m";

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <test_id>\n";
    return 1;
  }

  int test_id = 0;
  try {
    test_id = std::stoi(argv[1]);
  } catch (...) {
    std::cout << RED << "Invalid test id: " << argv[1] << RESET << "\n";
    return 2;
  }

  bool ok = false;
  if (test_id == 1) {
    try {
      ok = test_1();
    } catch (const std::exception &e) {
      std::cout << RED << "Test threw: " << e.what() << RESET << "\n";
      ok = false;
    }
  } else {
    std::cout << RED << "Unknown test id: " << test_id << RESET << "\n";
    return 2;
  }

  std::cout << (ok ? GREEN : RED) << (ok ? "PASSED" : "FAILED") << RESET
            << "\n";

  TestAPI api;
  std::vector<float> embedding = api.get_embeddings({100});
  float sum = 0;
  for (auto entry : embedding) {
    sum += entry;
  }
  std::cout << sum;

  return ok ? 0 : 3;
}