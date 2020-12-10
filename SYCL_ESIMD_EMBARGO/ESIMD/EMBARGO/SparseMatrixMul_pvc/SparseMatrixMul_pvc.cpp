//==-------- SparseMatrixMul_pvc.cpp  - DPC++ ESIMD on-device test -------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// TODO enable on Windows and Level Zero
// REQUIRES: linux && gpu && opencl
// RUN: %clangxx-esimd -fsycl %s -o %t.out
// RUNx: %ESIMD_RUN_PLACEHOLDER %t.out %S/band27-1m.dat 2

#include "../../esimd_test_utils.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <iostream>
#include <memory>
#include <string>

#include <CL/sycl.hpp>
#include <CL/sycl/INTEL/esimd.hpp>

#ifdef DUMP_ENABLE
#define DUMP(x) std::cout << x
#else
#define DUMP(x)
#endif

///////////////////////////////////////////////////////////////////////////////
// Configuration parameters for ESIMD host and kernel
///////////////////////////////////////////////////////////////////////////////
#define OWORD_BUF_ALIGNMENT (4) // Kernel required alignment for OWORD reads

// The following 3 parameters are for controlling max # of the sparse matrix
// rows processed per enqueue.
// Total number of active h/w threads
//   = MULTIPLIER * THREAD_SPACE_WIDTH
// Total number of sparse matrix rows processed
//   = MULTIPLIER * THREAD_SPACE_WIDTH * ROWS_PER_THREAD
// where
//   THREAD_SPACE_WIDTH corresponds to thread space width
//   MULTIPLIER corresponds to thread space height
//   ROWS_PER_THREAD corresponds to max scatter read capability
#define THREAD_SPACE_WIDTH (256)
#define MULTIPLIER (512)
#define ROWS_PER_THREAD (8)

#define ROUND(value, unit)                                                     \
  (((value) / (unit) + (((value) % (unit)) ? 1 : 0)) * (unit))
#define MIN(value1, value2) ((value1 < value2) ? value1 : value2)
#define MAX(value1, value2) ((value1 > value2) ? value1 : value2)
#define ABS(x) (((x) < 0) ? -(x) : (x))

#define SPARSE_FACTOR 10

// Structure for storing SparseMatrix in compressed sparse row format.
struct CsrSparseMatrix {
  unsigned num_rows;
  unsigned num_cols;
  unsigned num_nonzeros;
  unsigned *Arow; // pointer to the extents of rows for a CSR sparse matrix
  unsigned *Acol; // pointer to the column indices for a CSR sparse matrix
  float *Anz;     // pointer to the non-zero values for a CSR sparse matrix
  ~CsrSparseMatrix() {
    Arow = nullptr;
    Arow = nullptr;
    Acol = nullptr;
  }
};

using ushort = unsigned short;
using namespace cl::sycl;
using namespace sycl::INTEL::gpu;
using namespace std;

using IndexType = unsigned int;
using ValueType = float;

// template <typename IndexType, typename ValueType>
void SpmvCsr(ValueType *ANZ_BUF,  // CURBE  parameter
             IndexType *ACOL_BUF, // CURBE  parameter
             IndexType *AROW_BUF, // CURBE  parameter
             ValueType *X_BUF,    // CURBE  parameter
             ValueType *Y_BUF,    // CURBE  parameter
             IndexType row_start, // CURBE  parameter
             short row_stride,    // CURBE  parameter
             IndexType max_rows,  // CURBE  parameter
             IndexType *v_st_ptr, unsigned int gid0, unsigned int gid1);

int ReadCsrFile(const char *csr_filename, CsrSparseMatrix &csr, queue &q) {
  // This subroutine is to read in a CSR formatted matrix from a file
  // Param csr_filename: is an input file containing Spmv_csr.
  // Param csr: this structure will contain Spmv_csr after this call.

  auto dev = q.get_device();
  auto ctxt = q.get_context();

  FILE *f = fopen(csr_filename, "rb");
  if (f == NULL) {
    fprintf(stderr, "Error opening file %s", csr_filename);
    std::exit(1);
  }

  // Reads # cols (unsigned).
  if (fread(&csr.num_cols, sizeof(unsigned), 1, f) != 1) {
    fprintf(stderr, "Error reading num_cols from %s\n", csr_filename);
    std::exit(1);
  }
  fprintf(stderr, "csr.num_cols = %d\n", csr.num_cols);

  // Reads # rows (unsigned).
  if (fread(&csr.num_rows, sizeof(unsigned), 1, f) != 1) {
    fprintf(stderr, "Error reading num_rows from %s\n", csr_filename);
    std::exit(1);
  }
  fprintf(stderr, "csr.num_rows = %d\n", csr.num_rows);

  // Reads # non-zero values (unsigned).
  if (fread(&csr.num_nonzeros, sizeof(unsigned), 1, f) != 1) {
    fprintf(stderr, "Error reading num_nonzeros from %s\n", csr_filename);
    std::exit(1);
  }
  fprintf(stderr, "csr.num_nonzeros = %d\n", csr.num_nonzeros);

  // Reads column indices (unsigned *).
  // Do we need to aligned to 0x1000?
  csr.Acol =
      (unsigned *)malloc_shared(csr.num_nonzeros * sizeof(unsigned), dev, ctxt);
  if (fread(csr.Acol, sizeof(unsigned), csr.num_nonzeros, f) !=
      csr.num_nonzeros) {
    fprintf(stderr, "Error reading column indices from %s\n", csr_filename);
    std::exit(1);
  }
  for (unsigned int i = 0; i != csr.num_nonzeros; i++) {
    DUMP("Acol[" << i << "] = " << csr.Acol[i] << std::endl);
  }

  // Reads extent of rows (unsigned *).
  csr.Arow = (unsigned *)malloc_shared((csr.num_rows + 1) * sizeof(unsigned),
                                       dev, ctxt);
  if (fread(csr.Arow, sizeof(unsigned), csr.num_rows + 1, f) !=
      csr.num_rows + 1) {
    fprintf(stderr, "Error reading extent of rows from %s\n", csr_filename);
    std::exit(1);
  }
  for (unsigned int i = 0; i != csr.num_rows + 1; i++) {
    DUMP("Arow[" << i << "] = " << csr.Arow[i] << std::endl);
  }

  // Reads all non-zero values (float *).
  csr.Anz = (float *)malloc_shared(csr.num_nonzeros * sizeof(float), dev, ctxt);
  if (fread(csr.Anz, sizeof(float), csr.num_nonzeros, f) != csr.num_nonzeros) {
    fprintf(stderr, "Error reading non-zeros from %s\n", csr_filename);
    std::exit(1);
  }
  for (unsigned int i = 0; i != csr.num_nonzeros; i++) {
    DUMP("anz[" << i << "] = " << csr.Anz[i] << std::endl);
  }

  fclose(f);

  return 0;
}

#define OWORD_BUF_ALIGNMENT (4) // Required alignment for OWORD reads
#define THREAD_SPACE_WIDTH (256) // Thread space width
#define ROWS_PER_THREAD (8) // Process 8 row per thread

//////////////////////////////////////////////////////////////////////////////
// ESIMD kernel:
//    Generic SpMV kernel for CSR format.
//////////////////////////////////////////////////////////////////////////////

//#define ENABLE_PRINTF
template <class T>
void writeBuf(void *buf, const char *str, T val, uint32_t &idx) {
  if (!buf)
    return;

  uint strl = 0;
  while (*(str + strl) != 0) {
    *((unsigned char *)buf + idx) = *(str + strl);
    strl++;
    idx++;
  }

  // Terminate string
  *((unsigned char *)buf + idx) = 0;
  idx++;

  // Write sizeof val
  *((unsigned char *)buf + idx) = sizeof(T);
  idx++;

  // Write actual value
  auto addr = (T *)((unsigned char *)buf + idx);
  *addr = val;
  idx += sizeof(T);
}

#define SZ_PT 2000

ESIMD_INLINE void SpmvCsr(ValueType *ANZ_BUF,  // CURBE  parameter
                          IndexType *ACOL_BUF, // CURBE  parameter
                          IndexType *AROW_BUF, // CURBE  parameter
                          ValueType *X_BUF,    // CURBE  parameter
                          ValueType *Y_BUF,    // CURBE  parameter
                          IndexType row_start, // CURBE  parameter
                          short row_stride,    // CURBE  parameter
                          IndexType max_rows,  // CURBE  parameter
                          IndexType *v_st_ptr, unsigned int gid0,
                          unsigned int gid1, char *dbgBuf) {
  //--------------------------------------------------------------------
  // Read in the AROW vector with a stride calculated in v_st using a
  // scattered read. We do this in order to increase chances that
  // concurrently running threads access contiguous rows from the ANZ/ACOL
  // vector.
  //--------------------------------------------------------------------
  simd<IndexType, ROWS_PER_THREAD> v_ar;
  simd<IndexType, ROWS_PER_THREAD> v_sz;

#ifdef ENABLE_PRINTF
  constexpr int gid = 2000000;
  uint32_t idx = SZ_PT * gid0;
#endif

  auto v_st = block_load<IndexType, ROWS_PER_THREAD>(v_st_ptr);
  v_st *= sizeof(ValueType);

  int row_number =
      row_start + gid1 * THREAD_SPACE_WIDTH * ROWS_PER_THREAD + gid0;

  IndexType *baseAddr =
      (IndexType *)((char *)AROW_BUF +
                    (unsigned int)row_number * sizeof(ValueType));
  // Read arow(row_number)
  v_ar = gather<IndexType, ROWS_PER_THREAD>(baseAddr, v_st);

  simd<IndexType, ROWS_PER_THREAD> v_sz_st;

  // Read arow(row_number + 1)
  v_sz_st = v_st + sizeof(IndexType);
  baseAddr = (IndexType *)((char *)AROW_BUF +
                           (unsigned int)row_number * sizeof(ValueType));
  v_sz = gather<IndexType, ROWS_PER_THREAD>(baseAddr, v_sz_st);

  //--------------------------------------------------------------------
  // Compute sz = arow(row_number + 1) - arow(row_number)
  //--------------------------------------------------------------------
  v_sz = v_sz - v_ar;

  if (v_sz.any()) {
    //--------------------------------------------------------------------
    // Read in the Y vector with a stride calculated in v_st using a
    // scattered read for the same reason as for the AROW vector.
    //--------------------------------------------------------------------
    simd<uint, ROWS_PER_THREAD> v_y_dw;
    baseAddr = (IndexType *)((char *)Y_BUF +
                             (unsigned int)row_number * sizeof(ValueType));
    v_y_dw = gather<IndexType, ROWS_PER_THREAD>(baseAddr, v_st);

    //--------------------------------------------------------------------
    // Process all elements of one row at a time.
    //--------------------------------------------------------------------
    auto v_y = v_y_dw.format<ValueType>();

    for (ushort i = 0;
         (i < ROWS_PER_THREAD) && (row_number + i * row_stride < max_rows);
         i++) {
      IndexType row_begin = v_ar[i];
      IndexType row_length = v_sz[i];

      for (IndexType j = 0; j < row_length; j += 32) {
        IndexType row_slice_index = row_begin + j;
        IndexType row_slice_offset = row_slice_index * sizeof(ValueType);

        //--------------------------------------------------------------------
        // Read in the ANZ and ACOL vector using block reads. We process in the
        // ANZ and ACOL vector in slices of size 32, 16, 8 or 4 depending on the
        // row length.
        //--------------------------------------------------------------------
        IndexType row_remain_length = row_length - j;
        ushort row_slice_length =
            row_remain_length <= 32u ? row_remain_length : 32u;

        if (row_slice_length > 16) {
          simd<ValueType, 32> v_an;
          simd<uint, 32> v_ac;
          baseAddr = (IndexType *)((char *)ACOL_BUF + row_slice_offset);
          v_ac = block_load<uint, 32>(baseAddr);
          baseAddr = (IndexType *)((char *)ANZ_BUF + row_slice_offset);
          v_an = block_load<ValueType, 32>((ValueType *)baseAddr);

          //----------------------------------------------------------
          // We need to zero out the overfill portion of the final slice.
          //----------------------------------------------------------
          simd<ushort, 8> v_init8(0, 1);
          simd<ushort, 16> v_init16;
          simd<ushort, 8> row_slice_length_rep(row_slice_length);
          v_init16.select<8, 1>(0) = v_init8;
          v_init16.select<8, 1>(0) += 16;
          v_init16.select<8, 1>(8) = v_init16.select<8, 1>(0);
          v_init16.select<8, 1>(8) += 8;
          v_ac.select<8, 1>(16).merge(v_ac.select<8, 1>(16), 0,
                                      v_init16.select<8, 1>(0).read() <
                                          row_slice_length_rep);
          v_ac.select<8, 1>(24).merge(v_ac.select<8, 1>(24), 0,
                                      v_init16.select<8, 1>(8).read() <
                                          row_slice_length_rep);

          //--------------------------------------------------------------------
          // Read in the X vector using a scatter read and then perform the
          // actual compute.
          //--------------------------------------------------------------------
          simd<uint, 32> v_x_dw;
          v_x_dw.select<16, 1>(0) = gather<uint, 16>(
              (uint *)X_BUF, v_ac.select<16, 1>(0) * sizeof(ValueType));
          v_x_dw.select<16, 1>(16) = gather<uint, 16>(
              (uint *)X_BUF, v_ac.select<16, 1>(16) * sizeof(ValueType));
          auto v_x = v_x_dw.format<ValueType>();
          v_y.select<1, 1>(i) +=
              reduce<ValueType>(v_an * v_x.read(), std::plus<>());
        } else if (row_slice_length > 8) {
          simd<ValueType, 16> v_an;
          simd<uint, 16> v_ac;
          baseAddr = (IndexType *)((char *)ACOL_BUF + row_slice_offset);
          v_ac = block_load<uint, 16>(baseAddr);
          baseAddr = (IndexType *)((char *)ANZ_BUF + row_slice_offset);
          v_an = block_load<ValueType, 16>((ValueType *)baseAddr);

          //----------------------------------------------------------
          // We need to zero out the overfill portion of the final slice.
          //----------------------------------------------------------
          simd<ushort, 8> v_init8(0, 1);
          v_init8 += 8;
          v_ac.select<8, 1>(8).merge(v_ac.select<8, 1>(8), 0,
                                     v_init8 < row_slice_length);

          //-------------------------------------------------------------------------------
          // Read in the X vector using a scatter read and then perform the
          // actual compute.
          //-------------------------------------------------------------------------------
          simd<uint, 16> v_x_dw;
          v_x_dw.template select<16, 1>(0) = gather<uint, 16>(
              (uint *)X_BUF, v_ac.select<16, 1>(0) * sizeof(ValueType));
          auto v_x = v_x_dw.template format<ValueType>();
          v_y.select<1, 1>(i) += reduce<ValueType>(
              v_an.template select<16, 1>(0).read() * v_x.read(),
              std::plus<>());
        } else if (row_slice_length > 4) {
          simd<ValueType, 8> v_an;
          simd<uint, 8> v_ac;

          baseAddr = (IndexType *)((char *)ACOL_BUF + row_slice_offset);
          v_ac = block_load<uint, 8>(baseAddr);
          baseAddr = (IndexType *)((char *)ANZ_BUF + row_slice_offset);
          v_an = block_load<ValueType, 8>((ValueType *)baseAddr);

          //----------------------------------------------------------
          // We need to zero out the overfill portion of the final slice.
          //----------------------------------------------------------
          simd<ushort, 8> v_init8(0, 1);
          v_ac.merge(v_ac, 0, v_init8 < row_slice_length);

          //-------------------------------------------------------------------------------
          // Read in the X vector using a scatter read and then perform the
          // actual compute.
          //-------------------------------------------------------------------------------
          simd<uint, 8> v_x_dw;
          v_x_dw.template select<8, 1>(0) = gather(
              (uint *)X_BUF, v_ac.select<8, 1>(0).read() * sizeof(ValueType));
          auto v_x = v_x_dw.template select<8, 1>(0).format<ValueType>();
          v_y.select<1, 1>(i) += reduce<ValueType>(
              v_an.template select<8, 1>(0).read() * v_x.read(), std::plus<>());
        } else {
          simd<ValueType, 8> v_an;
          simd<uint, 8> v_ac;
          baseAddr = (IndexType *)((char *)ACOL_BUF + row_slice_offset);
          v_ac.template select<4, 1>(0) = block_load<uint, 4>(baseAddr);
          baseAddr = (IndexType *)((char *)ANZ_BUF + row_slice_offset);
          v_an.template select<4, 1>(0) =
              block_load<ValueType, 4>((ValueType *)baseAddr);

          //----------------------------------------------------------
          // We need to zero out the overfill portion of the final slice.
          //----------------------------------------------------------
          simd<ushort, 8> v_init8(0, 1);

          v_ac.merge(v_ac, 0, v_init8 < row_slice_length);

          //-------------------------------------------------------------------------------
          // Read in the X vector using a scatter read and then perform the
          // actual compute.
          //-------------------------------------------------------------------------------
          simd<uint, 8> v_x_dw;

          v_x_dw.template select<8, 1>(0) = gather(
              (uint *)X_BUF, v_ac.select<8, 1>(0).read() * sizeof(ValueType));
          auto v_x = v_x_dw.template select<4, 1>(0).format<ValueType>();
          v_y.select<1, 1>(i) += reduce<ValueType>(
              v_an.template select<4, 1>(0).read() * v_x.read(), std::plus<>());
        }
      }
    }

    //--------------------------------------------------------------------
    // Write out the Y vector with a stride calculated in v_st using a
    // scattered write.
    //--------------------------------------------------------------------
    baseAddr = (IndexType *)((char *)Y_BUF + row_number * sizeof(ValueType));

    scatter<ValueType, ROWS_PER_THREAD>((ValueType *)baseAddr, v_y, v_st);
  }
}

// return msecond
static double report_time(const string &msg, event e) {
  cl_ulong time_start =
      e.get_profiling_info<info::event_profiling::command_start>();
  cl_ulong time_end =
      e.get_profiling_info<info::event_profiling::command_end>();
  double elapsed = (time_end - time_start) / 1e6;
  // cerr << msg << elapsed << " msecs" << std::endl;
  return elapsed;
}

int RunCsrSpmvOnGpu(const CsrSparseMatrix &csr, int num_iter, queue &q) {
  // This subroutine is for multiplying a sparse matrix with a vector using
  // the GPU
  // Param csr: is an input sparse matrix stored in CSR format.
  // Equation to be performed is as follows:
  //   Y vector = Y vector + csr sparseMatrix * X vector
  // Before performing the above calculation, the subroutine will
  // 1. align the X, Y, csr dimensions to OWORD_BUF_ALIGNMENT
  // 2. initialize X and Y vectors with randomized values generated
  // using the same random seed.
  // In this example, this subroutine will do the above num_iter times with
  // the same initial X and Y vectors, and will compare first Y result with
  // subsequent Y (num_iter - 1) results.
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  srand(1);

  unsigned rounded_num_rows = ROUND(csr.num_rows, OWORD_BUF_ALIGNMENT);
  unsigned rounded_num_cols = ROUND(csr.num_cols + 1, OWORD_BUF_ALIGNMENT);

  // Randomizes x and y arrays.  They are aligned to OWORD_BUF_ALIGNMENT.
  float *x =
      (float *)malloc_shared(rounded_num_cols * sizeof(float), dev, ctxt);
  float *y =
      (float *)malloc_shared(rounded_num_rows * sizeof(float), dev, ctxt);

  x[0] = 0;

  for (unsigned i = 1; i < csr.num_cols + 1; i++) {
    x[i] = static_cast<float>(rand() / (RAND_MAX + 1.0));
  }

  for (unsigned i = csr.num_cols + 1; i < rounded_num_cols; i++) {
    x[i] = 0.0;
  }

  for (unsigned i = 0; i < csr.num_rows; i++) {
    y[i] = static_cast<float>(rand() / (RAND_MAX + 1.0));
  }

  for (unsigned i = csr.num_rows; i < rounded_num_rows; i++) {
    y[i] = 0.0;
  }

  // "ref" will contain reference value computed on CPU.
  float *ref = new float[rounded_num_rows];
  memcpy(ref, y, rounded_num_rows * sizeof(float));

  // compute on cpu
  for (unsigned i = 0; i < csr.num_rows; ++i) {
    DUMP("row" << i << " = " << ref[i] << std::endl);
    for (unsigned k = csr.Arow[i]; k < csr.Arow[i + 1]; ++k) {
      ref[i] += csr.Anz[k] * x[csr.Acol[k] + 1];
      DUMP("\t += " << csr.Anz[k] << " * " << x[csr.Acol[k] + 1] << std::endl);
    }
    DUMP("Ref row: " << i << ", ref[" << i << "] = " << ref[i] << std::endl);
  }

  // Creates aligned version of CSR arrays:
  //   anz for non-zeros
  //   arow for extent of rows
  //   acol for column indices
  // They are aligned to OWORD_BUF_ALIGNMENT.
  unsigned int rounded_anz_length = 0;
  for (unsigned i = 0; i < csr.num_rows; i++) {
    unsigned int row_length = csr.Arow[i + 1] - csr.Arow[i];
    rounded_anz_length += row_length;
  }

  unsigned int *arow = (unsigned int *)malloc_shared(
      (rounded_num_rows + 1) * sizeof(unsigned int), dev, ctxt);
  float *anz =
      (float *)malloc_shared(rounded_anz_length * sizeof(float), dev, ctxt);
  unsigned *acol = (unsigned *)malloc_shared(
      rounded_anz_length * sizeof(unsigned), dev, ctxt);

  arow[0] = 0;

  for (unsigned i = 0; i < csr.num_rows; i++) {
    unsigned row_start = csr.Arow[i];
    unsigned row_end = csr.Arow[i + 1];
    unsigned row_length = row_end - row_start;

    unsigned rounded_row_length = row_length;
    unsigned rounded_row_start = arow[i];
    unsigned rounded_row_end = rounded_row_start + rounded_row_length;

    arow[i + 1] = rounded_row_end;

    for (unsigned j = 0; j < row_length; j++) {
      anz[rounded_row_start] = csr.Anz[row_start];
      acol[rounded_row_start++] = csr.Acol[row_start++] + 1;
    }

    for (unsigned j = row_length; j < rounded_row_length; j++) {
      anz[rounded_row_start] = 0;
      acol[rounded_row_start++] = 0;
    }
  }

  for (unsigned i = csr.num_rows; i < rounded_num_rows; i++) {
    arow[i + 1] = arow[i];
  }

  // Creates num_iter copies of y vectors.
  float **y_vec =
      (float **)malloc_shared(num_iter * sizeof(float **), dev, ctxt);

  for (int i = 0; i < num_iter; i++) {
    y_vec[i] =
        (float *)malloc_shared(rounded_num_rows * sizeof(float), dev, ctxt);
    memcpy(y_vec[i], y, rounded_num_rows * sizeof(float));
  }

  // Setup additional kernel input data below.

  // The following 3 parameters defined before are for controlling
  // max # of the sparse matrix rows processed per enqueue.
  //   - THREAD_SPACE_WIDTH corresponds to thread space width
  //   - MULTIPLIER corresponds to thread space height
  //   - ROWS_PER_THREAD corresponds to max scatter read capability
  // v_st contains scattered read offset locations to relative rows per thread.
  unsigned *v_st =
      (unsigned *)malloc_shared(ROWS_PER_THREAD * sizeof(unsigned), dev, ctxt);
  for (int k = 0; k < ROWS_PER_THREAD; k++) {
    v_st[k] = k * THREAD_SPACE_WIDTH;
  }

  // Sets the loop-invariant kernel arguments for "SpmvCsr".
  // The first kernel argument is the non-zeros buffer index.
  // The second kernel argument is the column indices buffer index.
  // The third kernel argument is the extent of rows buffer index.
  // The fourth kernel argument is the X vector buffer index.
  // The fifth kernel argument is the Y vector buffer index, set in loop.
  // The sixth kernel argument indicates the start row of the input matrix
  // to be processed, set in loop.
  // The seventh kernel argument indicates the thread space width.
  // The eighth kernel argument indicates the max row of the input matrix.
  // The ninth kernel argument corresponds to the scattered read offset
  // locations, corresponding to which rows to be processed.
  short row_stride = THREAD_SPACE_WIDTH;
  unsigned max_rows = csr.num_rows;

  // Sets the loop-invariant kernel arguments for "SpmvCsrSimple".
  // The first kernel argument is the non-zeros buffer index.
  // The second kernel argument is the column indices buffer index.
  // The third kernel argument is the extent of rows buffer index.
  // The fourth kernel argument is the X vector buffer index.
  // The fifth kernel argument is the Y vector buffer index.
  // The sixth kernel argument indicates the start row of the input matrix
  // to be processed.

  // Total number of active h/w threads per enqueue
  //   = MULTIPLIER * THREAD_SPACE_WIDTH
  // Total number of spacse matrix rows processed per enqueue
  //   = MULTIPLIER * THREAD_SPACE_WIDTH * ROWS_PER_THREAD
  // batch_count indicates how many enqueues are needed to process the
  // input sparse matrix.
  unsigned batch_thread_count = THREAD_SPACE_WIDTH * MULTIPLIER;
  unsigned batch_row_size = batch_thread_count * ROWS_PER_THREAD;
  unsigned batch_count = ROUND(csr.num_rows, batch_row_size) / batch_row_size;

  // An event, "sync_event", is created to track the status of the task.
  // Will be used with enqueue.

  // Adds a CmKernel pointer to CmTask.
  // This task has one kernel, "kernel_spmv_csr".

  // The number of rows in a matrix may not be a multiple of batch_row_size.
  // Calculates the last batch's actual required batch_thread_count.
  // This value will be stored in last_batch_thread_count.
  unsigned last_batch_thread_count = 0;
  if ((csr.num_rows / batch_row_size) * batch_row_size < csr.num_rows) {
    bool done = false;
    for (unsigned int k = 0; k < MULTIPLIER && done == false; k++) {
      unsigned batch_row_start =
          (csr.num_rows / batch_row_size) * batch_row_size;
      for (unsigned int j = 0; j < THREAD_SPACE_WIDTH; j++) {
        unsigned thread_start_row_index =
            batch_row_start + k * THREAD_SPACE_WIDTH * ROWS_PER_THREAD + j;
        if (thread_start_row_index < csr.num_rows) {
          last_batch_thread_count++;
        } else {
          done = true;
          break;
        }
      }
    }
  } else {
    last_batch_thread_count = batch_thread_count;
  }

  last_batch_thread_count = ROUND(last_batch_thread_count, THREAD_SPACE_WIDTH);

#if 1
  std::cout << "batch_count = " << batch_count << std::endl;
  std::cout << "batch_row_size = " << batch_row_size << std::endl;
  std::cout << "last_batch_thread_count = " << last_batch_thread_count
            << std::endl;
#endif

  unsigned thread_count = 0;
  double exec_start, exec_stop;
  float exec_total = 0.0f;

#ifndef ENABLE_PRINTF
  constexpr unsigned int SZ = 0;
  char *dbgBuf = nullptr;
#else
  constexpr unsigned int SZ = 200000;
  char *dbgBuf = (char *)malloc_shared(SZ * sizeof(uint), dev, ctxt);
#endif
  memset(dbgBuf, 0, SZ * sizeof(uint));

  for (int i = 0; i < num_iter; i++) {
    if (i == 1) {
      // exec_start = getTimeStamp();
    }

    // Does the Y = Y + csr * X num_iter times through the "SpmvCsr" kernel.
    for (unsigned batch_row_start = 0; batch_row_start < csr.num_rows;
         batch_row_start += batch_row_size) {

      if (batch_row_start + batch_row_size < csr.num_rows) {
        thread_count = batch_thread_count;
      } else {
        thread_count = last_batch_thread_count;
      }

      unsigned int total_threads = thread_count;
      auto GroupRange = cl::sycl::range<2>(THREAD_SPACE_WIDTH,
                                           thread_count / THREAD_SPACE_WIDTH);
      auto TaskRange = cl::sycl::range<2>(1, 1);

      auto e = q.submit([&](cl::sycl::handler &cgh) {
        cgh.parallel_for<class spmv_csr>(
            GroupRange * TaskRange, [=](item<2> it) SYCL_ESIMD_KERNEL {
              using namespace sycl::INTEL::gpu;
              SpmvCsr(anz, acol, arow, x, y_vec[i], batch_row_start, row_stride,
                      max_rows, v_st, it.get_id(0), it.get_id(1),
                      i == 0 ? dbgBuf : nullptr);
            });
      });
      e.wait();
    }
  }

#ifdef ENABLE_PRINTF
  std::cout << "Dbgbuf:" << std::endl;
  char tmp[SZ + 1];
  uint idx1 = 0;
  uint totalSize = 0;
  bool strName = true;
  for (unsigned int i = 0; i < SZ; i++) {
    if (strName) {
      tmp[idx1++] = dbgBuf[i];
      if (dbgBuf[i] == 0) {
        // end string
        strName = false;
        continue;
      }
    } else {
      // read actual value
      unsigned int size = dbgBuf[i++];
      totalSize += size;
      if (size > 0) {
        std::cout << (const char *)tmp << "(" << size << " bytes) = ";
        if (size > 1000)
          std::cout << "***** Bad size *****" << std::endl;
        else {

          for (unsigned int j = 0; j != size; j++, i++) {
            printf("0x%x ", ((unsigned char *)dbgBuf)[i]);
          }
          std::cout << std::endl;
        }
        // i--;
      }
      for (unsigned int i1 = 0; i1 != 512; i1++)
        tmp[i1] = 0;
      strName = true;
      idx1 = 0;
    }
  }
#endif

  std::cout << std::endl;

  unsigned int errorcount = 0;
  constexpr int maxErrors = 10;
  for (unsigned int i = 0; i != csr.num_rows; i++) {
    float *res = y_vec[1];
    float rel_error = ABS(ABS(ref[i]) - ABS(res[i]));
    if (rel_error > 0x002) {
      std::cout << "row#" << i << ": ref = " << ref[i]
                << ", res = " << y_vec[1][i] << "(" << y_vec[1][i] / ref[i]
                << "x)" << std::endl;
      errorcount++;
    }
    if (errorcount > maxErrors) {
      std::cout << "Truncating at " << maxErrors << " errors" << std::endl;
      break;
    }
  }

  for (int i = 1; i < num_iter; i++) {
    // Compares Y[i] vectors with Y[0] vector.
    unsigned error_count = 0;
    float max_rel_error = 0;
    unsigned int error_index = 0;
    float error_ref = 0;
    float error_res = 0;
    // float *ref = y_vec[0]; WHY??
    float *res = y_vec[i];
    for (unsigned int j = 0; j < csr.num_rows; j++) {
      float rel_error = ABS(ref[j] - res[j]) / MAX(ref[j], res[j]);
      if (max_rel_error < rel_error) {
        max_rel_error = rel_error;
        error_index = j;
        error_ref = ref[j];
        error_res = res[j];
        error_count++;
      }
    }

    std::cout << "error_count " << error_count << ", max_rel_error "
              << max_rel_error << std::endl;
    if (max_rel_error > 0.002) {
      std::cout << "ERROR: Discrepency in run " << i << "!" << std::endl;
      std::cout << "Max rel error = " << max_rel_error << std::endl;
      std::cout << "Error index = " << error_index << std::endl;
      std::cout << "Error ref = " << error_ref << std::endl;
      std::cout << "Error res = " << error_res << std::endl;
      std::cout << "Error count = " << error_count << std::endl;
    }
  }

  int count = num_iter - 1;
  // std::cout << "Kernel execution time is " << (total_time / 1000000.0f /
  // count) << " msec" << std::endl;
  std::cout << "Total time is " << (1000.0f * exec_total / count) << " msec"
            << std::endl;
  std::cout << "Total UPIteration count is " << count << std::endl;
  std::cout << "batch_row_size is " << (batch_row_size) << std::endl;
  std::cout << "csr.num_rows is " << (csr.num_rows) << std::endl;

  memcpy(y, y_vec[0], rounded_num_rows * sizeof(float));

  float *res = y;

  // Compares reference value with final Y value.
  float max_rel_error = 0;
  float abs_error = 0;
  unsigned int error_index = 0;
  float error_ref = 0;
  float error_res = 0;

  for (unsigned int i = 0; i < csr.num_rows; i++) {
    float rel_error = ABS(ref[i] - res[i]) / MAX(ref[i], res[i]);
    if (max_rel_error < rel_error) {
      max_rel_error = rel_error;
      error_index = i;
      error_ref = ref[i];
      error_res = res[i];
      abs_error = ABS(ref[i] - res[i]);
    }
  }

  if (max_rel_error > 0.02 && abs_error > 0.000005) {
    std::cout << "Max rel error = " << max_rel_error << std::endl;
    std::cout << "Error index = " << error_index << std::endl;
    std::cout << "Error ref = " << error_ref << std::endl;
    std::cout << "Error res = " << error_res << std::endl;
    std::cout << "FAILED" << std::endl;
    return 1;
  } else {
    std::cout << "Result matches reference CPU implementation" << std::endl;
    std::cout << "PASSED" << std::endl;
    return 0;
  }
  delete[] ref;
  sycl::free(res);

  sycl::free(x, ctxt);
  sycl::free(y, ctxt);
  sycl::free(anz, ctxt);
  sycl::free(arow, ctxt);
  sycl::free(acol, ctxt);
  for (int i = 0; i < num_iter; i++) {
    sycl::free(y_vec[i], ctxt);
  }
  sycl::free(y_vec, ctxt);
  sycl::free(v_st, ctxt);
  sycl::free(dbgBuf, ctxt);
  sycl::free(csr.Acol, ctxt);
  sycl::free(csr.Arow, ctxt);
  sycl::free(csr.Anz, ctxt);
}

int main(int argc, char *argv[]) {
  // This program shows the usage of a kernel in one task to perform
  // sparse multiplication using the GPU.
  // The equation to be performed is as follows:
  //   Y = Y + [sparse matrix] * X vector
  // The above equation is performed through this core subroutine:
  //   float *RunCsrSpmvOnGpu(const CsrSparseMatrix &csr)
  // This subroutine takes an input of a sparse matrix with csr format.
  // Internally it will initialize the X and Y vectors with pseudo random
  // numbers generated using 1 as the seed value.
  // RunCsrSpmvOnGpu(csr) compares the final Y vector with the
  // reference values on cpu.

  // By default, use "Protein_csr.dat" as input sparse matrix with csr format.
  const char *csr_filename = "Protein_csr.dat";
  int num_iter = 0;
  if (argc == 3) {
    csr_filename = argv[1];
    num_iter = atoi(argv[2]) + 1;
  } else {
    std::cerr << "Unknown option. Exiting..." << std::endl;
    std::cerr << "Usage: SparseMatrixMul.exe input_file iteration_count"
              << std::endl;
    std::exit(1);
  }

  queue q(esimd_test::ESIMDSelector{}, esimd_test::createExceptionHandler(),
          property::queue::enable_profiling{});

  CsrSparseMatrix csr;
  ReadCsrFile(csr_filename, csr, q);

  std::cout << "Using " << csr.num_rows << "-by-" << csr.num_cols
            << " matrix with " << csr.num_nonzeros << " nonzero values"
            << std::endl;

  return RunCsrSpmvOnGpu(csr, num_iter, q);
}
