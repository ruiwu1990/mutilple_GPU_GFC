/*
Copyright (c) 2011, Texas State University-San Marcos. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted for academic, research, experimental, or personal use provided
that the following conditions are met:

   * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
   * Redistributions in binary form must reproduce the above copyright notice,
     this list of conditions and the following disclaimer in the documentation
     and/or other materials provided with the distribution.
   * Neither the name of Texas State University-San Marcos nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

For all other uses, please contact the Office for Commercialization and Industry
Relations at Texas State University-San Marcos <http://www.txstate.edu/ocir/>.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Molly A. O'Neil and Martin Burtscher
*/


#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#include <iostream>
#include <fstream>

#include "book.h"
#include "fileSeperatorMerger.h"

#define ull unsigned long long
#define MAX (64*1024*1024)

#define WARPSIZE 32
#define MAXWARP 32

__constant__ int dimensionalityd; // dimensionality parameter
__constant__ ull *cbufd; // ptr to uncompressed data
__constant__ unsigned char *dbufd; // ptr to compressed data
__constant__ ull *fbufd; // ptr to decompressed data
__constant__ int *cutd; // ptr to chunk boundaries
__constant__ int *offd; // ptr to chunk offsets after compression

/************************************************************************************/

/*
This is the GPU compression kernel, which should be launched using the block count
and warps/block:
  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();

Inputs
------
dimensionalityd: dimensionality of trace (from cmd line)
cbufd: ptr to the uncompressed data
cutd: ptr to array of chunk boundaries

Output
------
The compressed data, in dbufd 
Compressed chunk offsets for offset table, in offd
*/

/* adapted from Hacker's Delight */
__device__ int clzll(ull x)
{
  int n;
  if (x == 0) return(64);

  n = 0;
  if (x <= 0x00000000FFFFFFFFL) {n = n + 32; x = x << 32;}
  if (x <= 0x0000FFFFFFFFFFFFL) {n = n + 16; x = x << 16;}
  if (x <= 0x00FFFFFFFFFFFFFFL) {n = n + 8; x = x << 8;}
  if (x <= 0x0FFFFFFFFFFFFFFFL) {n = n + 4; x = x << 4;}
  if (x <= 0x3FFFFFFFFFFFFFFFL) {n = n + 2; x = x << 2;}
  if (x <= 0x7FFFFFFFFFFFFFFFL) {n = n + 1;}
  
  return n;
}

__global__ void CompressionKernel()
{
  register int offset, code, bcount, tmp, off, beg, end, lane, warp, iindex, lastidx, start, term;
  register ull diff, prev;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)]; // shared space for prefix sum

  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to compress
  start = 0;
  if (warp > 0) start = cutd[warp-1];
  term = cutd[warp];
  off = ((start+1)/2*17);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {
    // calculate delta between value to compress and prediction
    // and negate if negative
    diff = cbufd[i] - prev;
    code = (diff >> 60) & 8;
    if (code != 0) {
      diff = -diff;
    }

    // count leading zeros in positive delta
    //bcount = 8 - (__clzll(diff) >> 3);
	/*
	* without __clzll
	*/
	bcount = 8 - (clzll(diff) >> 3);

    if (bcount == 2) bcount = 3; // encode 6 lead-zero bytes as 5

    // prefix sum to determine start positions of non-zero delta bytes
    ibufs[iindex] = bcount;
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-1];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-2];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-4];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-8];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-16];
    __threadfence_block();

    // write out non-zero bytes of delta to compressed buffer
    beg = off + (WARPSIZE/2) + ibufs[iindex-1];
    end = beg + bcount;
    for (; beg < end; beg++) {
      dbufd[beg] = diff;
      diff >>= 8;
    }

    if (bcount >= 3) bcount--; // adjust byte count for the dropped encoding
    tmp = ibufs[lastidx];
    code |= bcount;
    ibufs[iindex] = code;
    __threadfence_block();

    // write out half-bytes of sign and leading-zero-byte count (every other thread
    // writes its half-byte and neighbor's half-byte)
    if ((lane & 1) != 0) {
      dbufd[off + (lane >> 1)] = ibufs[iindex-1] | (code << 4);
    }
    off += tmp + (WARPSIZE/2);

    // save prediction value from this subchunk (based on provided dimensionality)
    // for use in next subchunk
    prev = cbufd[i + offset];
  }

  // save final value of off, which is total bytes of compressed output for this chunk
  if (lane == 31) offd[warp] = off;
}

/************************************************************************************/

/*
This is the GPU decompression kernel, which should be launched using the block count
and warps/block:
  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();

Inputs
------
dimensionalityd: dimensionality of trace
dbufd: ptr to array of compressed data
cutd: ptr to array of chunk boundaries

Output
------
The decompressed data in fbufd
*/

__global__ void DecompressionKernel()
{
  register int offset, code, bcount, off, beg, end, lane, warp, iindex, lastidx, start, term;
  register ull diff, prev;
  __shared__ int ibufs[32 * (3 * WARPSIZE / 2)];

  // index within this warp
  lane = threadIdx.x & 31;
  // index within shared prefix sum array
  iindex = threadIdx.x / WARPSIZE * (3 * WARPSIZE / 2) + lane;
  ibufs[iindex] = 0;
  iindex += WARPSIZE / 2;
  lastidx = (threadIdx.x / WARPSIZE + 1) * (3 * WARPSIZE / 2) - 1;
  // warp id
  warp = (threadIdx.x + blockIdx.x * blockDim.x) / WARPSIZE;
  // prediction index within previous subchunk
  offset = WARPSIZE - (dimensionalityd - lane % dimensionalityd) - lane;

  // determine start and end of chunk to decompress
  start = 0;
  if (warp > 0) start = cutd[warp-1];
  term = cutd[warp];
  off = ((start+1)/2*17);

  prev = 0;
  for (int i = start + lane; i < term; i += WARPSIZE) {

    // read in half-bytes of size and leading-zero count information
    if ((lane & 1) == 0) {
      code = dbufd[off + (lane >> 1)];
      ibufs[iindex] = code;
      ibufs[iindex + 1] = code >> 4;
    }
/*
/////////////////////////
//code = dbufd[off + (lane >> 1)]*((-1)*(lane&1)+1) + code*(lane&1);
//ibufs[iindex] = ((-1)*(lane&1)+1)*code + ibufs[iindex]*(lane&1);
    if ((lane & 1) == 0) {
//      code = dbufd[off + (lane >> 1)];
//      ibufs[iindex] = code;
      ibufs[iindex + 1] = code >> 4;
    }
*/	

//	ibufs[iindex + 1] = (code >> 4)*((-1)*(lane&1)+1) + ibufs[iindex + 1]*(lane&1);

    off += (WARPSIZE/2);
    __threadfence_block();
    code = ibufs[iindex];

    bcount = code & 7;
/////////////
    if (bcount >= 2) bcount++;
//bcount = ((bcount-2)>>31)*(-1)+1+bcount; 

    // calculate start positions of compressed data
    ibufs[iindex] = bcount;
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-1];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-2];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-4];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-8];
    __threadfence_block();
    ibufs[iindex] += ibufs[iindex-16];
    __threadfence_block();

    // read in compressed data (the non-zero bytes)
    beg = off + ibufs[iindex-1];
    off += ibufs[lastidx];
    end = beg + bcount - 1;
    diff = 0;
    for (; beg <= end; end--) {
      diff <<= 8;
      diff |= dbufd[end];
    }

//we can remove this if by using this/////////////////////////
//(-0.25*(code&8)+1)*diff;

    // negate delta if sign bit indicates it was negated during compression
    if ((code & 8) != 0) {
      diff = -diff;
    }

//diff=(-0.25*(code&8)+1)*diff;

    // write out the uncompressed word
    fbufd[i] = prev + diff;
    __threadfence_block();

    // save prediction for next subchunk
    prev = fbufd[i + offset];
  }
}

/************************************************************************************/

static void CudaTest(char *msg)
{
  cudaError_t e;

  cudaThreadSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "%s: %d\n", msg, e);
    fprintf(stderr, "%s\n", cudaGetErrorString(e));
    exit(-1);
  }
}

//this function is used to get the length of one file
int getFileLength(FILE* inputFile)
{
  int size=-1;

  if (inputFile==NULL)
  {
	std::cout<<"error open file"<<std::endl;
	exit(0);
  }
  else
  {
    fseek (inputFile, 0, SEEK_END);   // non-portable
    size=ftell (inputFile);
    rewind (inputFile);
    return size;
  }
}

/************************************************************************************/

static void Compress(int blocks, int warpsperblock, int dimensionality, FILE * resultFile, FILE * inPutFile, float & mediantime)
{

  cudaGetLastError();  // reset error value

  // allocate CPU buffers
  //ull is unsigned long long int
  ull *cbuf = (ull *)malloc(sizeof(ull) * MAX); // uncompressed data
  if (cbuf == NULL) {
    fprintf(stderr, "cannot allocate cbuf\n"); exit(-1);
  }
  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data
  if (dbuf == NULL) {
    fprintf(stderr, "cannot allocate dbuf\n"); exit(-1);
  }
  int *cut = (int *)malloc(sizeof(int) * blocks * warpsperblock); // chunk boundaries
  if (cut == NULL) {
    fprintf(stderr, "cannot allocate cut\n"); exit(-1);
  }
  int *off = (int *)malloc(sizeof(int) * blocks * warpsperblock); // offset table
  if (off == NULL) {
    fprintf(stderr, "cannot allocate off\n"); exit(-1);
  }

  // read in trace to cbuf
  int doubles = fread(cbuf, 8, MAX, inPutFile);
  //int fileSize = getFileLength(inPutFile);
//std::cout<<"file size is "<<fileSize<<std::endl;
  //int doubles = fread(cbuf, 8, fileSize, inPutFile);

  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock);
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0, before = 0, d = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
    curr += per;
    cut[i] = min(curr, doubles);
    if (cut[i] - before > 0) {
      d = cut[i] - before;
    }
    before = cut[i];
  }

  // set the pad values to ensure correct prediction
  if (d <= WARPSIZE) {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = 0;
    }
  } else {
    for (int i = doubles - padding; i < doubles; i++) {
      cbuf[i] = cbuf[(i & -WARPSIZE) - (dimensionality - i % dimensionality)];
    }
  }

  // allocate GPU buffers
  ull *cbufl; // uncompressed data
  char *dbufl; // compressed data
  int *cutl; // chunk boundaries
  int *offl; // offset table
  if (cudaSuccess != cudaMalloc((void **)&cbufl, sizeof(ull) * doubles))
    fprintf(stderr, "could not allocate cbufd\n");
  CudaTest("couldn't allocate cbufd");
  if (cudaSuccess != cudaMalloc((void **)&dbufl, sizeof(char) * ((doubles+1)/2*17)))
    fprintf(stderr, "could not allocate dbufd\n");
  CudaTest("couldn't allocate dbufd");
  if (cudaSuccess != cudaMalloc((void **)&cutl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate cutd\n");
  CudaTest("couldn't allocate cutd");
  if (cudaSuccess != cudaMalloc((void **)&offl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate offd\n");
  CudaTest("couldn't allocate offd");

  // copy buffer starting addresses (pointers) and values to constant memory
  if (cudaSuccess != cudaMemcpyToSymbol(dimensionalityd, &dimensionality, sizeof(int)))
    fprintf(stderr, "copying of dimensionality to device failed\n");
  CudaTest("dimensionality copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(cbufd, &cbufl, sizeof(void *)))
    fprintf(stderr, "copying of cbufl to device failed\n");
  CudaTest("cbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dbufd, &dbufl, sizeof(void *)))
    fprintf(stderr, "copying of dbufl to device failed\n");
  CudaTest("dbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(cutd, &cutl, sizeof(void *)))
    fprintf(stderr, "copying of cutl to device failed\n");
  CudaTest("cutl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(offd, &offl, sizeof(void *)))
    fprintf(stderr, "copying of offl to device failed\n");
  CudaTest("offl copy to device failed");

  // copy CPU buffer contents to GPU
  if (cudaSuccess != cudaMemcpy(cbufl, cbuf, sizeof(ull) * doubles, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cbuf to device failed\n");
  CudaTest("cbuf copy to device failed");
  if (cudaSuccess != cudaMemcpy(cutl, cut, sizeof(int) * blocks * warpsperblock, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cut to device failed\n");
  CudaTest("cut copy to device failed");


  //record time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord( start, 0 );

  CompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();
  CudaTest("compression kernel launch failed");

  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

  std::cout<<"compression time: "<<elapsedTime<<" millisecond"<<std::endl;
  mediantime=elapsedTime + mediantime;

  // transfer offsets back to CPU
  if(cudaSuccess != cudaMemcpy(off, offl, sizeof(int) * blocks * warpsperblock, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of off from device failed\n");
  CudaTest("off copy from device failed");

  // output header
  int num;
  int doublecnt = doubles-padding;
std::cout<<"compress block: "<<blocks<<std::endl;  
////////////////
  num = fwrite(&blocks, 4, 1, resultFile);
  assert(1 == num);
  num = fwrite(&warpsperblock, 1, 1, resultFile);
  assert(1 == num);
  num = fwrite(&dimensionality, 1, 1, resultFile);
  assert(1 == num);
  num = fwrite(&doublecnt, 4, 1, resultFile);
  assert(1 == num);
  // output offset table
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int start = 0;
    if(i > 0) start = cut[i-1];
    off[i] -= ((start+1)/2*17);
    num = fwrite(&off[i], 4, 1, resultFile); // chunk's compressed size in bytes
    assert(1 == num);
  }
  // output compressed data by chunk
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int offset, start = 0;
    if(i > 0) start = cut[i-1];
    offset = ((start+1)/2*17);
    // transfer compressed data back to CPU by chunk
    if (cudaSuccess != cudaMemcpy(dbuf + offset, dbufl + offset, sizeof(char) * off[i], cudaMemcpyDeviceToHost))
      fprintf(stderr, "copying of dbuf from device failed\n");
    CudaTest("dbuf copy from device failed");
    num = fwrite(&dbuf[offset], 1, off[i], resultFile);
    assert(off[i] == num);
  }

  free(cbuf);
  free(dbuf);
  free(cut);
  free(off);

  // Cleanup in the event of success.
  cudaEventDestroy( start );
  cudaEventDestroy( end );

  if (cudaSuccess != cudaFree(cbufl))
    fprintf(stderr, "could not deallocate cbufd\n");
  CudaTest("couldn't deallocate cbufd");
  if (cudaSuccess != cudaFree(dbufl))
    fprintf(stderr, "could not deallocate dbufd\n");
  CudaTest("couldn't deallocate dbufd");
  if (cudaSuccess != cudaFree(cutl))
    fprintf(stderr, "could not deallocate cutd\n");
  CudaTest("couldn't deallocate cutd");
  if (cudaSuccess != cudaFree(offl))
    fprintf(stderr, "could not deallocate offd\n");
  CudaTest("couldn't deallocate offd");
}

/************************************************************************************/

static void Decompress(int blocks, int warpsperblock, int dimensionality, int doubles, FILE * resultFile, FILE * inPutFile, float & mediantimeD)
{
  cudaGetLastError();  // reset error value

  // allocate CPU buffers

  char *dbuf = (char *)malloc(sizeof(char) * ((MAX+1)/2*17)); // compressed data, divided by chunk
  if (dbuf == NULL) { 
    fprintf(stderr, "cannot allocate dbuf\n"); exit(-1); 
  }
  ull *fbuf = (ull *)malloc(sizeof(ull) * MAX); // decompressed data
  if (fbuf == NULL) { 
    fprintf(stderr, "cannot allocate fbuf\n"); exit(-1);
  }
  int *cut = (int *)malloc(sizeof(int) * blocks * warpsperblock); // chunk boundaries
  if (cut == NULL) { 
    fprintf(stderr, "cannot allocate cut\n"); exit(-1);
  }
  int *off = (int *)malloc(sizeof(int) * blocks * warpsperblock); // offset table
  if(off == NULL) {
    fprintf(stderr, "cannot allocate off\n"); exit(-1);
  }

  // read in offset table
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int num = fread(&off[i], 4, 1, inPutFile);
    assert(1 == num);
  }

  // calculate required padding for last chunk
  int padding = ((doubles + WARPSIZE - 1) & -WARPSIZE) - doubles;
  doubles += padding;

  // determine chunk assignments per warp
  int per = (doubles + blocks * warpsperblock - 1) / (blocks * warpsperblock); 
  if (per < WARPSIZE) per = WARPSIZE;
  per = (per + WARPSIZE - 1) & -WARPSIZE;
  int curr = 0;
  for (int i = 0; i < blocks * warpsperblock; i++) {
    curr += per;
    cut[i] = min(curr, doubles);
  }

  // allocate GPU buffers
  char *dbufl; // compressed data
  ull *fbufl; // uncompressed data
  int *cutl; // chunk boundaries
  if (cudaSuccess != cudaMalloc((void **)&dbufl, sizeof(char) * ((doubles+1)/2*17)))
    fprintf(stderr, "could not allocate dbufd\n");
  CudaTest("couldn't allocate dbufd");
  if (cudaSuccess != cudaMalloc((void **)&fbufl, sizeof(ull) * doubles))
    fprintf(stderr, "could not allocate fbufd\n");
  CudaTest("couldn't allocate fbufd");
  if (cudaSuccess != cudaMalloc((void **)&cutl, sizeof(int) * blocks * warpsperblock))
    fprintf(stderr, "could not allocate cutd\n");
  CudaTest("couldn't allocate cutd");

  // copy buffer starting addresses (pointers) and values to constant memory
  if (cudaSuccess != cudaMemcpyToSymbol(dimensionalityd, &dimensionality, sizeof(int))) 
    fprintf(stderr, "copying of dimensionality to device failed\n");
  CudaTest("dimensionality copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(dbufd, &dbufl, sizeof(void *)))
    fprintf(stderr, "copying of dbufl to device failed\n");
  CudaTest("dbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(fbufd, &fbufl, sizeof(void *)))
    fprintf(stderr, "copying of fbufl to device failed\n");
  CudaTest("fbufl copy to device failed");
  if (cudaSuccess != cudaMemcpyToSymbol(cutd, &cutl, sizeof(void *)))
    fprintf(stderr, "copying of cutl to device failed\n");
  CudaTest("cutl copy to device failed");

  // read in input data and divide into chunks
  for(int i = 0; i < blocks * warpsperblock; i++) {
    int num, chbeg, start = 0;
    if (i > 0) start = cut[i-1];

    chbeg = ((start+1)/2*17);
    // read in this chunk of data (based on offsets)
    num = fread(&dbuf[chbeg], 1, off[i], inPutFile);
    assert(off[i] == num);
    // transfer the chunk to the GPU

    if (cudaSuccess != cudaMemcpy(dbufl + chbeg, dbuf + chbeg, sizeof(char) * off[i], cudaMemcpyHostToDevice)) 
      fprintf(stderr, "copying of dbuf to device failed\n");
    CudaTest("dbuf copy to device failed");
  }

  // copy CPU cut buffer contents to GPU
  if (cudaSuccess != cudaMemcpy(cutl, cut, sizeof(int) * blocks * warpsperblock, cudaMemcpyHostToDevice))
    fprintf(stderr, "copying of cut to device failed\n");
  CudaTest("cut copy to device failed");

  //record time
  cudaEvent_t start, end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);

  cudaEventRecord( start, 0 );

  DecompressionKernel<<<blocks, WARPSIZE*warpsperblock>>>();
  CudaTest("decompression kernel launch failed");

  cudaEventRecord( end, 0 );
  cudaEventSynchronize( end );

  float elapsedTime;
  cudaEventElapsedTime( &elapsedTime, start, end );

 std::cout<<"decompression time: "<<elapsedTime<<" millisecond"<<std::endl;
 // timeResultFile<<elapsedTime<<"	"; 
  mediantimeD = elapsedTime + mediantimeD;   


  // transfer result back to CPU
  if (cudaSuccess != cudaMemcpy(fbuf, fbufl, sizeof(ull) * doubles, cudaMemcpyDeviceToHost))
    fprintf(stderr, "copying of fbuf from device failed\n");
  CudaTest("fbuf copy from device failed");

  // output decompressed data
  int num = fwrite(fbuf, 8, doubles-padding, resultFile);
  assert(num == doubles-padding);

  free(dbuf);
  free(fbuf);
  free(cut);

 

  if(cudaSuccess != cudaFree(dbufl))
    fprintf(stderr, "could not deallocate dbufd\n");
  CudaTest("couldn't deallocate dbufd");
  if(cudaSuccess != cudaFree(fbufl))
    fprintf(stderr, "could not deallocate fbufl\n");
  CudaTest("couldn't deallocate fbufl");
  if(cudaSuccess != cudaFree(cutl))
    fprintf(stderr, "could not deallocate cutd\n");
  CudaTest("couldn't deallocate cutd");
}

/************************************************************************************/

static int VerifySystemParameters()
{
  assert(1 == sizeof(char));
  assert(4 == sizeof(int));
  assert(8 == sizeof(ull));
  int val = 1;
  assert(1 == *((char *)&val));

  int current_device = 0, sm_per_multiproc = 0; 
  int max_compute_perf = 0, max_perf_device = 0; 
  int device_count = 0, best_SM_arch = 0; 
  int arch_cores_sm[3] = { 1, 8, 32 }; 
  cudaDeviceProp deviceProp; 

  cudaGetDeviceCount(&device_count);
  if (device_count == 0) {
    fprintf(stderr, "There is no device supporting CUDA\n");
    exit(-1);
  }
   
  // Find the best major SM Architecture GPU device 
  for (current_device = 0; current_device < device_count; current_device++) { 
    cudaGetDeviceProperties(&deviceProp, current_device);
    if (deviceProp.major > 0 && deviceProp.major < 9999) { 
      best_SM_arch = max(best_SM_arch, deviceProp.major); 
    }
  }
   
  // Find the best CUDA capable GPU device 
  for (current_device = 0; current_device < device_count; current_device++) { 
    cudaGetDeviceProperties(&deviceProp, current_device); 
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {
      sm_per_multiproc = 1;
    } 
    else if (deviceProp.major <= 2) { 
      sm_per_multiproc = arch_cores_sm[deviceProp.major]; 
    } 
    else { // Device has SM major > 2 
      sm_per_multiproc = arch_cores_sm[2]; 
    }
      
    int compute_perf = deviceProp.multiProcessorCount * 
                       sm_per_multiproc * deviceProp.clockRate; 
      
    if (compute_perf > max_compute_perf) { 
      // If we find GPU of SM major > 2, search only these 
      if (best_SM_arch > 2) { 
        // If device==best_SM_arch, choose this, or else pass 
        if (deviceProp.major == best_SM_arch) { 
          max_compute_perf = compute_perf; 
          max_perf_device = current_device; 
        } 
      } 
      else { 
        max_compute_perf = compute_perf; 
        max_perf_device = current_device; 
      } 
    } 
  } 
   
  cudaGetDeviceProperties(&deviceProp, max_perf_device); 
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {
    fprintf(stderr, "There is no CUDA capable  device\n");
    exit(-1);
  }
  if (deviceProp.major < 2) {
    fprintf(stderr, "Need at least compute capability 2.0\n");
    exit(-1);
  }
  if (deviceProp.warpSize != WARPSIZE) {
    fprintf(stderr, "Warp size must be %d\n", deviceProp.warpSize);
    exit(-1);
  }
  if ((WARPSIZE <= 0) || (WARPSIZE & (WARPSIZE-1) != 0)) {
    fprintf(stderr, "Warp size must be greater than zero and a power of two\n");
    exit(-1);
  }

  return max_perf_device;
}

/*
this function is to get mena of tempArray
*/
void getMedian(float* tempArray, int length)
{
  int j = 0;
  float tmp = 0;
  for(int i=0;i<length;i++){
    j = i;
    for(int k = i;k<length;k++){
      if(tempArray[j]>tempArray[k]){
        j = k;
      }
    }
    tmp = tempArray[i];
    tempArray[i] = tempArray[j];
    tempArray[j] = tmp;
  }
}

//multiGPU
/*
build file name
*/
char* buildFileName(char* inputFileName, int appendNum)
{
	std::string fileName;
	fileName.clear();
	fileName.append(inputFileName);
	fileName.append(".");
	char intBuf[10];
	sprintf(intBuf,"%d",appendNum);
	fileName.append(intBuf);

	char* resultFileName = new char[fileName.size() + 1];
	std::copy(fileName.begin(), fileName.end(), resultFileName);
	resultFileName[fileName.size()] = '\0';

	return resultFileName;
}

/*
This struct is used to import data into different threads
*/
struct DataStruct
{
	//compression part
	int deviceID;
	int blocks; 
	int warpsperblock; 
	int dimensionality;
	FILE * resultFile;
	FILE * inPutFile;
	float mediantime;
};
struct DataStructD
{
	//decompression part
	int deviceIDD;
	int blocksD;
	int warpsperblockD;
	int dimensionalityD;
	int doublesD;
	FILE * resultFileD;
	FILE * inPutFileD;
	float mediantimeD;
};
//this funcion is for thread operation
void* routine(void *pvoidData)
{
	DataStruct *data = (DataStruct*)pvoidData;
	cudaSetDevice(data->deviceID);
	Compress(data->blocks, data->warpsperblock, data->dimensionality, data->resultFile, data->inPutFile, data->mediantime); 
	return 0;
}

void* routineD(void *pvoidData)
{
	DataStructD *data = (DataStructD*)pvoidData;
	cudaSetDevice(data->deviceIDD);
	Decompress(data->blocksD, data->warpsperblockD, data->dimensionalityD, data->doublesD, data->resultFileD, data->inPutFileD, data->mediantimeD); 
	return 0;
}

/************************************************************************************/

int main(int argc, char *argv[])
{
  int blocks, warpsperblock, dimensionality;
  int device;

  fprintf(stderr, "GPU FP Compressor v2.1\n");

  device = VerifySystemParameters();
  cudaSetDevice(device);

  cudaFuncSetCacheConfig(CompressionKernel, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(DecompressionKernel, cudaFuncCachePreferL1);

  //output result
  //multiGPU
  char * resultFile;
  char * inPutFile;
  
  //output time results 
  std::ofstream timeResultFile;
  std::ofstream timeResultFileD;



  if((6 == argc) || (7 == argc)) 
  { /* compress */
    if(6 == argc) {
      dimensionality = 1;
    } else {
      dimensionality = atoi(argv[1]);
    }
	if(6 == argc) {
	  timeResultFile.open(argv[2]);
	  timeResultFileD.open(argv[5]);
    } else {
	  timeResultFile.open(argv[3]);
	  timeResultFileD.open(argv[6]);
    }
    //assert((0 < dimensionality) && (dimensionality <= WARPSIZE));
//set up for loop here
	int num, doubles;

	float mediantime[12];
	float mediantimeD[12];

	//multiGPU
	int numGPU;
	cudaGetDeviceCount(&numGPU);
	DataStruct * tempData= new DataStruct[numGPU];
	DataStructD * tempDataD=new DataStructD[numGPU];
	CUTThread * thread = new CUTThread[numGPU];
	std::cout<<"There are "<<numGPU<<" GPU in this machine"<<std::endl;

	char** tempSeperateInputFile = new char*[numGPU];
	char** tempSeperateOutputFile = new char*[numGPU];
	FILE** tempInputFile = new FILE*[numGPU];
	FILE** tempOutputFile = new FILE*[numGPU];
	
	//1.41421 is sqrt(2), (tempLoop*1.41421)+1 is to get the roof value
//	for(double tempLoop = 1; tempLoop <=1024 ; tempLoop=(int)(tempLoop*1.41421)+1)
	for(double tempLoop = 100; tempLoop <=100 ; tempLoop=(int)(tempLoop*1.41421)+1)
    {			
		//initialize mediantime and mediantimeD
		for(int mediantimeCount = 0; mediantimeCount < 12 ; mediantimeCount++)
		{
			mediantime[mediantimeCount] = 0;
			mediantimeD[mediantimeCount] = 0;
		}
		for(int tempWLoop = 1; tempWLoop<=11 ;tempWLoop+=1)
//		for(int tempWLoop = 30; tempWLoop<=32 ;tempWLoop++)
		{

			if(6 == argc) 
			{
				//multiGPU
				resultFile = argv[1];	
				inPutFile = argv[3];

			}
			else 
			{
				//multiGPU
     			resultFile = argv[2];
				inPutFile = argv[4];
			}

			blocks = tempLoop;
			warpsperblock = MAXWARP;
		
			//multiGPU
			std::ifstream fileStream;
			fileStream.open(inPutFile, std::ios::in | std::ios::binary);
			int fileSize = getFileSize(&fileStream);
//			float chunkSize = (float)fileSize / (numGPU-1);//if we have 3 GPUs, divided by 2	
float chunkSize = (float)fileSize / (numGPU);//if we have 3 GPUs, divided by 2	
			chunkFile(inPutFile,inPutFile,chunkSize);		
			//multiGPU
			for(int tempGPULoop = 1 ; tempGPULoop <= numGPU ; tempGPULoop++)
			{
				//create right name for each file
				tempSeperateInputFile[tempGPULoop-1] = buildFileName(inPutFile, tempGPULoop);
				tempSeperateOutputFile[tempGPULoop-1] = buildFileName(resultFile, tempGPULoop);


//std::cout<<"this is "<<tempSeperateInputFile[tempGPULoop-1]<<" input file"<<std::endl;
//std::cout<<"this is "<<tempSeperateOutputFile[tempGPULoop-1]<<" output file"<<std::endl;
				//open files
				tempInputFile[tempGPULoop-1] = fopen(tempSeperateInputFile[tempGPULoop-1],"rb");

				tempOutputFile[tempGPULoop-1] = fopen(tempSeperateOutputFile[tempGPULoop-1],"wb");
			
				tempData[tempGPULoop-1].deviceID = tempGPULoop-1;
				tempData[tempGPULoop-1].blocks = blocks;
				tempData[tempGPULoop-1].warpsperblock = warpsperblock;
				tempData[tempGPULoop-1].dimensionality = dimensionality;
				tempData[tempGPULoop-1].resultFile = tempOutputFile[tempGPULoop-1];
				tempData[tempGPULoop-1].inPutFile = tempInputFile[tempGPULoop-1];
				tempData[tempGPULoop-1].mediantime = mediantime[tempWLoop];

				thread[tempGPULoop-1] = start_thread(routine, &tempData[tempGPULoop-1]);

			}
			for(int tempGPULoop = 1 ; tempGPULoop <= numGPU ; tempGPULoop++)
			{
				end_thread( thread[tempGPULoop-1] );	
	
				mediantime[tempWLoop] += tempData[tempGPULoop-1].mediantime;
std::cout<<tempData[tempGPULoop-1].mediantime<<" time"<<std::endl;
			
//std::cout<<"this is "<<tempGPULoop<<"finished"<<std::endl;
				fclose(tempInputFile[tempGPULoop-1]); 
				fclose(tempOutputFile[tempGPULoop-1]);
			}

		    //assert(0 == fread(&dummy, 1, 1, inPutFile));

			//decompress process here 

			if(5 == argc) 
			{
				inPutFile = argv[1];
				resultFile = argv[4];
			}
			else
			{
				inPutFile = argv[2];
				resultFile = argv[5];
			}

			//multiGPU
			for(int tempGPULoop = 1 ; tempGPULoop <= numGPU ; tempGPULoop++)
			{
				//create right name for each file
				tempSeperateInputFile[tempGPULoop-1] = buildFileName(inPutFile, tempGPULoop);
				tempSeperateOutputFile[tempGPULoop-1] = buildFileName(resultFile, tempGPULoop);
				//open files
				tempInputFile[tempGPULoop-1] = fopen(tempSeperateInputFile[tempGPULoop-1],"rb");
				tempOutputFile[tempGPULoop-1] = fopen(tempSeperateOutputFile[tempGPULoop-1],"wb");


				num = fread(&blocks, 4, 1, tempInputFile[tempGPULoop-1]);
				assert(1 == num);
				blocks &= 255;
				num = fread(&warpsperblock, 1, 1, tempInputFile[tempGPULoop-1]);
				assert(1 == num);
				warpsperblock &= 255;
				num = fread(&dimensionality, 1, 1, tempInputFile[tempGPULoop-1]);
				assert(1 == num);
				dimensionality &= 255;
				num = fread(&doubles, 4, 1, tempInputFile[tempGPULoop-1]);
				assert(1 == num);

				blocks = tempLoop;

				
				tempDataD[tempGPULoop-1].deviceIDD = tempGPULoop-1;
				tempDataD[tempGPULoop-1].blocksD = blocks;
				tempDataD[tempGPULoop-1].warpsperblockD = warpsperblock;
				tempDataD[tempGPULoop-1].dimensionalityD = dimensionality;
				tempDataD[tempGPULoop-1].doublesD = doubles;
				tempDataD[tempGPULoop-1].resultFileD = tempOutputFile[tempGPULoop-1];
				tempDataD[tempGPULoop-1].inPutFileD = tempInputFile[tempGPULoop-1];
				tempDataD[tempGPULoop-1].mediantimeD = mediantimeD[tempWLoop];

				thread[tempGPULoop-1] = start_thread(routineD, &tempDataD[tempGPULoop-1]);

			}
			for(int tempGPULoop = 1 ; tempGPULoop <= numGPU ; tempGPULoop++)
			{
				end_thread( thread[tempGPULoop-1] );	
	
				mediantimeD[tempWLoop] += tempDataD[tempGPULoop-1].mediantimeD;
			
//std::cout<<"this is "<<tempGPULoop<<"finished"<<std::endl;
				fclose(tempInputFile[tempGPULoop-1]); 
				fclose(tempOutputFile[tempGPULoop-1]);
			}

		}

		//multiGPU
		//combine all the result files into one file
		//this is for compressed file		
		if(6 == argc) 
		{
			//multiGPU
			joinFile(argv[1], argv[1]);
		}
		else 
		{
			//multiGPU
			joinFile(argv[2], argv[2]);
		}
		//this is for decompressed file
		if(5 == argc) 
		{	
			joinFile(argv[4], argv[4]);
		}
		else
		{
			joinFile(argv[5], argv[5]);
		}

		//this is the median, in fact just sort the array
		getMedian(mediantime, 11);
for(int i=1; i<12; i++)
	std::cout<<mediantime[i]<<std::endl;
	std::cout<<"the median is "<<mediantime[6]<<std::endl;
		timeResultFile<<mediantime[6];
		timeResultFile<<std::endl;
		//this is the mediantime 
		getMedian(mediantimeD, 12);

		timeResultFileD<<mediantimeD[6];
		timeResultFileD<<std::endl;

    }
  delete [] thread;
  delete [] tempSeperateInputFile;
  delete [] tempSeperateOutputFile;
  delete [] tempInputFile;
  delete [] tempOutputFile;

  }

  else {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "compress: %s blocks warps/block (dimensionality) < file.in > file.gfc\n", argv[0]);
    fprintf(stderr, "decompress: %s < file.gfc > file.out\n", argv[0]);
  }

//  fclose(resultFile); 
  return 0;
}

