#include <stdio.h>
#include <string>
#include <assert.h>

#include "simple-interop.hxx"
#include "render.hxx"

struct global_state_t {
  simple_interop::simple_interop_t interop;
  raytracer_t raytracer;

  global_state_t(uint width, uint height, uint nsamples) :
    interop(width,height),
    raytracer(width,height, nsamples) {}
};

global_state_t* global_state;

void render() {
  unsigned int width = global_state->interop.width, height = global_state->interop.height;

  global_state->interop.cuda_render([=](uchar3 *output) {
      global_state->raytracer.draw_scene(output, width, height);
    });

  
  glFlush();
  glutPostRedisplay();  
}

void handle_keyboard(int key, int x, int y) {
  auto & camera = global_state->raytracer.camera;
  bool moved = true;
  
  switch(key) {
  case GLUT_KEY_UP:
    camera.move(camera.forward);
    break;

  case GLUT_KEY_DOWN:
    camera.move(0-camera.forward);
    break;

  case GLUT_KEY_LEFT:
    camera.rotate(camera.left);
    break;

  case GLUT_KEY_RIGHT:
    camera.rotate(0-camera.left);
    break;

  default:
    moved = false;
    break;  
  }

  if(moved)
    global_state->raytracer.camera_moved();
  
  glutPostRedisplay();
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(2000,2000);
  glutCreateWindow("Render with CUDA");

  global_state_t main_global_state(1000,1000, 8);
  global_state = &main_global_state;
  
  glutDisplayFunc(render);
  glutSpecialFunc(handle_keyboard);
  glutMainLoop();

  printf("Did I get here?");
  return 0;
}
