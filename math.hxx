#pragma once

#include "helper_math.h"
#include "types.hxx"

namespace clt_math {

  __device__ float3 sample_cosine_weighted_direction(float3 normal, float r1, float r2) {
    float p1 = 2*M_PI*r1, p2 = r2, p2s = sqrt(r2);
    float3 w = normal, u = normalize(cross(fabs(w.x)>.1?float3{0,1} : float3{1}, w)),
      v = cross(w,u);
    float3 d = u*cos(p1)*p2s + v * sin(p1)*p2s + w*sqrt(1-p2);
    return normalize(d);
  }

  float3 rotate_xz(float3 vec, float angle) {
    float3 r1{cos(angle),  0,  sin(angle)};
    float3 r2{0,       1,  0};
    float3 r3{-sin(angle), 0,  cos(angle)};    

    float3 answer = {dot(r1,vec),
                     dot(r2,vec),
                     dot(r3,vec)};
    return normalize(answer); // it's a unitary transform, so get rid of the normalize
  }

  __device__ intersection_result_t intersect(const ray_t & ray,
                                             const sphere_t & sphere) {
    auto center = sphere.center;
    auto radius = sphere.radius;
    float3 op = center-ray.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0 
    float t, eps=2e-2, b=dot(op, ray.direction), det=b*b - dot(op,op) + radius*radius;
    
    if (det<0) return intersection_result_t{-1}; else det=sqrt(det); 
    float closest = (t=b-det)>eps ? t : ((t=b+det)>eps ? t : -1);

    if(closest < 0)
      return intersection_result_t{-1};

    float3 point = ray.origin + closest*ray.direction;
    float3 normal = (point - center)/radius;

    // Cornell box has us inside the spheres, so make sure to return a
    // normal in the appropriate direction: away from the incoming ray
    
    if(dot(ray.direction, normal) > 0)
      normal = -normal; 
    
    return intersection_result_t{closest,
        point+normal*eps,
        normal};
  }    
}
