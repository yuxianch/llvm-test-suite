#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#define INC1(x) ((x) = (x) + 1);

#define INC10(x)                                                               \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)                                                                      \
  INC1(x)

#define INC100(x)                                                              \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)                                                                     \
  INC10(x)

#define INC1000(x)                                                             \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)                                                                    \
  INC100(x)

#define INC10000(x)                                                            \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)                                                                   \
  INC1000(x)

#define INC100000(x)                                                           \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)                                                                  \
  INC10000(x)

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
class Inc;
template <class Kernel> void check_build_time(cl::sycl::queue &q) {
  cl::sycl::program program(q.get_context());
  auto start = std::chrono::steady_clock::now();
  program.build_with_kernel_type<Kernel>();
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed build time: " << elapsed_seconds.count() << "s\n";
}
int main(int argc, char **argv) {
  auto start = std::chrono::steady_clock::now();
  // Test program and kernel APIs when building a kernel.
  {
    cl::sycl::queue q;
    check_build_time<Inc>(q);

    int data = 0;
    {
      cl::sycl::buffer<int, 1> buf(&data, cl::sycl::range<1>(1));
      cl::sycl::range<1> NumOfWorkItems{buf.get_count()};

      q.submit([&](cl::sycl::handler &cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
        cgh.parallel_for<class Inc>(
            NumOfWorkItems, [=](cl::sycl::id<1> WIid) { TARGET_IMAGE(acc[0]) });
      });
    }
    // check_build_time<Inc>(q);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "elapsed kernel time: " << elapsed_seconds.count() << "s\n";
  }
}
