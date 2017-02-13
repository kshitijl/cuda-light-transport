#pragma once

#include <cuda.h>

using uchar = unsigned char;
using uint = unsigned int;

struct ray_t {
  float3 origin;
  float3 direction;
};  

struct intersection_result_t {
  float distance;
  float3 intersection_point;
  float3 surface_normal;
};

struct sphere_t {
  float radius;  
  float3 center;
  float3 emittance;
  float3 color;
};
