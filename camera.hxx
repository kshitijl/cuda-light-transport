#pragma once

#include "math.hxx"

struct camera_t {
  const float3 forward{0,0.0,8};
  const float left = 0.1;
  
  float3 eye{50,52,210.6};
  float zoom = 0.80;
  float3 original_direction = normalize(float3{0.0,-0.042612,-1});

  float3 direction = original_direction;

  float t = 0;
    
  void move(float3 way) {
    eye += way.z*direction;
  }

  void rotate(float amount) {
    t += amount;
    direction = clt_math::rotate_xz(original_direction, t);
  }    
};
