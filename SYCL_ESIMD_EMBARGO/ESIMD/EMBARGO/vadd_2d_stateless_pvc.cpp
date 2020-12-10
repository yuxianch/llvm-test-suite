//==------- vadd_2d_stateless_pvc.cpp  - DPC++ ESIMD on-device test --------==//
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

#include "esimd_test_utils.hpp"

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>
#include <iostream>

using namespace cl::sycl;

using namespace sycl::INTEL::gpu;

#define WIDTH 16
#define HEIGHT 8

int main(void) {
  constexpr unsigned Size = WIDTH * HEIGHT;

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler());

  auto dev = q.get_device();
  std::cout << "Running on " << dev.get_info<info::device::name>() << "\n";

  auto ctxt = q.get_context();
  float *A =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *B =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));
  float *C =
      static_cast<float *>(malloc_shared(Size * sizeof(float), dev, ctxt));

  for (unsigned i = 0; i < Size; ++i) {
    A[i] = B[i] = i;
    C[i] = 0.0f;
  }

  cl::sycl::range<1> GroupRange{1};
  cl::sycl::range<1> TaskRange{1};
  cl::sycl::nd_range<1> Range(GroupRange, TaskRange);

  auto e = q.submit([&](handler &cgh) {
    cgh.parallel_for<class Test>(Range, [=](nd_item<1> ndi) SYCL_ESIMD_KERNEL {
      using namespace sycl::INTEL::gpu;
      int i = ndi.get_global_id(0);

      constexpr unsigned NROW = WIDTH;
      constexpr unsigned NCOL = HEIGHT;
      constexpr unsigned VL = NROW * NCOL;

      constexpr unsigned BLKX = NROW;
      constexpr unsigned BLKY = NCOL;
      constexpr unsigned width = NROW * sizeof(float) - 1;
      constexpr unsigned height = NCOL - 1;
      constexpr unsigned pitch = width;
      constexpr unsigned xoff = 0;
      constexpr unsigned yoff = 0;

      simd<float, VL> va =
          esimd_2d_statelss_load<float, BLKX, BLKY, 1, false, false,
                                 CacheHint::Streaming, CacheHint::Uncached>(
              A, width, height, pitch, xoff, yoff);
      simd<float, VL> vb =
          esimd_2d_statelss_load<float, BLKX, BLKY, 1, false, false,
                                 CacheHint::Streaming, CacheHint::Uncached>(
              B, width, height, pitch, xoff, yoff);

      simd<float, VL> vc = va + vb;

      esimd_2d_statelss_store<float, BLKX, BLKY, CacheHint::Uncached,
                              CacheHint::WriteBack>(C, width, height, pitch,
                                                    xoff, yoff, vc);
    });
  });
  e.wait();

  int err_cnt = 0;

  for (unsigned i = 0; i < Size; ++i) {
    if (A[i] + B[i] != C[i]) {
      if (++err_cnt < 10) {
        std::cout << "failed at index " << i << ", " << C[i] << " != " << A[i]
                  << " + " << B[i] << "\n";
      }
    }
  }
  if (err_cnt > 0) {
    std::cout << "  pass rate: "
              << ((float)(Size - err_cnt) / (float)Size) * 100.0f << "% ("
              << (Size - err_cnt) << "/" << Size << ")\n";
  }

  std::cout << (err_cnt > 0 ? "FAILED\n" : "Passed\n");
  return err_cnt > 0 ? 1 : 0;
}
