#include "types.hxx"

#include <curand-done-right/curanddr.hxx>

__global__ void sample_paths(float3 *image, uint width, uint height,
                             uint nsamples,
                             uint frame_number,
                             camera_t camera,
                             const sphere_t* spheres, int nspheres) {
  uint x = blockIdx.x * blockDim.x + threadIdx.x;
  uint y = blockIdx.y * blockDim.y + threadIdx.y;
  uint sample = blockIdx.z * blockDim.z + threadIdx.z;

  if(x < width && y < height) {
    auto rr = curanddr::uniforms<2>(uint3{x,y,sample},
                                    uint2{frame_number,0});
    float2 ru = float2{rr[0], rr[1]};
    
    float3 camdir = camera.direction;;
    float3 cx = normalize(cross(float3{0, width*1.0f/height}, -camdir))*camera.zoom
      , cy = normalize(cross(cx, camdir))*camera.zoom;
    float3 direction = cx*( (ru.x + x-0.5f)/width - 0.5) +
      cy * ( (ru.y + 1.0f*y - 0.5)/height - 0.5) + camdir;
      
    ray_t ray{camera.eye, normalize(direction)};

    float3 accumulator{0,0,0}, mask{1,1,1};

    for(uint bounces = 0; bounces < 3; ++bounces) {
      intersection_result_t best{1e150};
      int best_sphere_i = 0;
    
      for(int ii = 0; ii < nspheres; ++ii) {
        auto result = clt_math::intersect(ray, spheres[ii]);
        if(result.distance > 0) {
          if(result.distance < best.distance) {
            best = result;
            best_sphere_i = ii;
          }
        }
      }
    
      if(best.distance < 1e150) {
        auto rx = curanddr::uniforms<2>(uint3{x,y,sample},
                                        uint2{frame_number, bounces+1});
        float2 random_uniforms = float2{rx[0], rx[1]};
        
        float3 new_dir = clt_math::sample_cosine_weighted_direction(best.surface_normal,
                                                                    random_uniforms.x,
                                                                    random_uniforms.y);
        ray = ray_t{best.intersection_point, new_dir};

        accumulator += mask*spheres[best_sphere_i].emittance;
        mask *= spheres[best_sphere_i].color;
      }
      else {
        break;
      }
    }

    int idx = (y*width + x)*nsamples + sample;

    image[idx] = accumulator/nsamples;
  }
}
