#include "types.hxx"

#include <curand-done-right/curanddr.hxx>

__device__ ray_t generate_camera_ray(const camera_t & camera,
                                     uint x, uint y, uint width, uint height,
                                     float rand1, float rand2) {
  float3 camdir = camera.direction;
  float3 cx = normalize(cross(float3{0, width*1.0f/height}, -camdir))*camera.zoom
    , cy = normalize(cross(cx, camdir))*camera.zoom;
  float3 direction = cx*( (rand1 + x-0.5f)/width - 0.5) +
    cy * ( (rand2 + 1.0f*y - 0.5)/height - 0.5) + camdir;
      
  return ray_t{camera.eye, normalize(direction)};
}

struct sample_t {
  uint pixel;  
  float3 weight;
  intersection_result_t last_hit;
};

__device__ int intersect_spheres(const ray_t & ray,
                                 const sphere_t * spheres, uint nspheres,
                                 intersection_result_t & result) {
  intersection_result_t best{1e150};
  int best_sphere_i = -1;
    
  for(int ii = 0; ii < nspheres; ++ii) {
    auto result = clt_math::intersect(ray, spheres[ii]);
    if(result.distance > 0) {
      if(result.distance < best.distance) {
        best = result;
        best_sphere_i = ii;
      }
    }
  }

  result = best;
  return best_sphere_i;
}

__global__ void sample_paths(float3 *image, uint width, uint height,
                             uint nsamples,
                             uint frame_number,
                             camera_t camera,
                             const sphere_t* spheres, int nspheres,
                             uint light_id) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint sample = blockIdx.z * blockDim.z + threadIdx.z;

  if(x < width && y < height) {
    auto rr = curanddr::uniforms<4>(uint3{x,y,sample},
                                    uint2{frame_number,0});
    auto ray = generate_camera_ray(camera, x, y, width, height,
                                   rr[0], rr[1]);
 
    float3 accumulator{0,0,0}, mask{1,1,1};

    auto direct_illumination = [&](intersection_result_t from,
                                   float rand1, float rand2) {
      float3 light_point = clt_math::sample_point_on_sphere(spheres[light_id],
                                                            rand1, rand2);
      float3 new_dir = light_point - from.intersection_point;
      
      ray_t shadow_ray{from.intersection_point, normalize(new_dir)};

      intersection_result_t ir;
      if(intersect_spheres(shadow_ray, spheres, nspheres, ir) == light_id) {
        // subtended angle weight
        float d= ir.distance, r = spheres[light_id].radius;
        float weight = 1 - sqrt(fmaxf(0,d*d - r*r))/d;
        return spheres[light_id].emittance * weight;;
      }
      else {
        return float3{0,0,0};
      }        
    };
    
    mgpu::iterate<3>([&](uint index) {
        intersection_result_t best;
        int best_sphere_i = intersect_spheres(ray, spheres, nspheres, best);
    
        if(best_sphere_i >= 0) {
          mask *= spheres[best_sphere_i].color;

          auto randoms = curanddr::uniforms<4>(uint3{x,y,sample},
                                               uint2{frame_number,index+1});
          accumulator += mask*direct_illumination(best,
                                                  randoms[0], randoms[1]);

          float3 new_dir = clt_math::sample_uniform_direction(best.surface_normal,
                                                              randoms[2], randoms[3]);
          
          ray = ray_t{best.intersection_point, normalize(new_dir)};
        }
      });
        
    
    int idx = (y*width + x)*nsamples + sample;
    image[idx] = accumulator/nsamples;
  }
}
