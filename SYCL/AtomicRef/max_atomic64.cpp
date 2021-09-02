// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include "max.h"
#include <iostream>
using namespace sycl;

int main() {
  queue q;

  if (!q.get_device().has(aspect::atomic64)) {
    std::cout << "Skipping test\n";
    return 0;
  }

  constexpr int N = 32;
  max_test<double>(q, N);

  // Include long tests if they are 64 bits wide
  if constexpr (sizeof(long) == 8) {
    max_test<long>(q, N);
    max_test<unsigned long>(q, N);
  }

  // Include long long tests if they are 64 bits wide
  if constexpr (sizeof(long long) == 8) {
    max_test<long long>(q, N);
    max_test<unsigned long long>(q, N);
  }

  std::cout << "Test passed." << std::endl;
}
