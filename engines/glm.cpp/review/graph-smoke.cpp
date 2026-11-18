#include "../src/rt.h"
#include <sycl/ext/oneapi/experimental/graph.hpp>
#include <chrono>
#include <cstdio>
using namespace glm;
namespace ex=sycl::ext::oneapi::experimental;
int main() {
 auto &d=Dev::get(); auto &q=d.q;
 printf("graph aspect=%d limited=%d\n",d.dev.has(sycl::aspect::ext_oneapi_graph),d.dev.has(sycl::aspect::ext_oneapi_limited_graph));
 auto *p=sycl::malloc_device<float>(4096,q);q.memset(p,0,4096*4).wait();
 try {
  ex::command_graph g{q};g.begin_recording(q);
  for(int n=0;n<32;n++) q.parallel_for(sycl::range<1>(4096),[=](sycl::id<1> i){p[i]+=1;});
  g.end_recording(q);auto e=g.finalize();
  for(int n=0;n<10;n++)q.ext_oneapi_graph(e);q.wait();
  auto st=std::chrono::steady_clock::now();
  for(int n=0;n<200;n++)q.ext_oneapi_graph(e);q.wait();
  double us=std::chrono::duration<double>(std::chrono::steady_clock::now()-st).count()*1e6/200;
  printf("32-kernel graph replay %.3f us, amortized %.3f us/kernel\n",us,us/32);
  float val;q.memcpy(&val,p,4).wait();printf("result %.0f expected %d\n",val,210*32);
 } catch(std::exception &e) {fprintf(stderr,"graph error %s\n",e.what());return 1;}
}
