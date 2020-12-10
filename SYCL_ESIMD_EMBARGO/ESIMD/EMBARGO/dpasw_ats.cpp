//==------------- dpasw_ats.cpp  - DPC++ ESIMD on-device test --------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUNx: %ESIMD_RUN_PLACEHOLDER %t.out

#include "../esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

int main(void) {
  constexpr unsigned Size = 64;
  constexpr unsigned VL = 16;
  constexpr unsigned GroupSize = 2;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";
  auto ctxt = q.get_context();

  int *C = static_cast<int *>(malloc_shared(Size * sizeof(int), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    C[i] = 0;
  }

  // We need that many task groups
  cl::sycl::range<1> GroupRange{2};

  // We need that many tasks in each group
  cl::sycl::range<1> TaskRange{GroupSize};
  cl::sycl::nd_range<1> Range(GroupRange, TaskRange);

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
      using namespace sycl::INTEL::gpu;

      simd<char, 8 * 16> va(0);
      auto ma = va.format<char, 8, 16>();
      ma.select<2, 1, 4, 4>(0, 0) = 4;

      simd<char, 8 * 8> vb(0);
      auto mb = vb.format<char, 8, 8>();
      mb.select<4, 2, 1, 1>(0, 0) = 4;

      simd<int, 8 * 8> vc(0);
      vc = esimd_dpasw<ESIMD_PRECISION_S2, ESIMD_PRECISION_S2, 8, 8, int, int,
                       int, 64, 32, 16>(vc, ma.format<int>(), mb.format<int>());

      for (int i = 0; i < 64; i += VL) {
        block_store<int, VL>(C + i, vc.select<VL, 1>(i));
      }
    });
  });
  e.wait();
  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (C[i] != 1) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << 1
                  << "\n";
      }
    }
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
