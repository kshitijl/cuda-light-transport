#include <iostream>
#include <iomanip>
#include <ctime>
#include <sstream>
#include <string>
#include <assert.h>

#include "simple-interop.hxx"
#include "write-ppm.hxx"

#include "render.hxx"

#include <GL/freeglut_ext.h>

struct global_state_t {
  uint width, height;
  simple_interop::simple_interop_t interop;
  raytracer_t raytracer;

  global_state_t(uint width, uint height, uint nsamples) :
    width(width), height(height),
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

void screenshot() {
  uint width = global_state->width, height = global_state->height;
  char * buffer = (char *)malloc(width*height*3);
  cudaMemcpy(buffer, global_state->interop.d_textureBufferData,
             width*height*3, cudaMemcpyDeviceToHost);

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);
  std::ostringstream oss;
  oss << std::put_time(&tm, "renders/%d-%m-%y-%H-%M-%S.ppm");
  auto str = oss.str();
  write_ppm(str.c_str(), width, height, buffer);
  free(buffer);
}

void handle_keyboard_special(int key, int x, int y) {
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

void handle_keyboard(uchar key, int x, int y) {
  switch(key) {
  case 27:
    glutLeaveMainLoop();
    break;

  case 'p':
    screenshot();
    break;

  default:
    break;
  }
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(2000,2000);
  glutCreateWindow("Render with CUDA");

  uint width = 1000, height = 1000;
  global_state_t main_global_state(width, height, 4);
  global_state = &main_global_state;
  
  glutDisplayFunc(render);
  glutSpecialFunc(handle_keyboard_special);
  glutKeyboardFunc(handle_keyboard);

  glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,
                GLUT_ACTION_GLUTMAINLOOP_RETURNS);
  
  glutMainLoop();

  screenshot();
  return 0;
}
