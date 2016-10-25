#include <GL/gl.h>
#include <GL/glut.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <time.h>

int width = 1024, height = 768;
GLuint mtexture;

void render(){
  glClear(GL_COLOR_BUFFER_BIT);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  int tt = clock() % 1000000;
  float angle = 360.0*tt/1000000;
  glRotatef(angle, 0, 0, 1);

  glEnable(GL_TEXTURE_2D); // you should use shader, but for an example fixed pipeline is ok ;)
  glBindTexture(GL_TEXTURE_2D, mtexture);

  float x = 0.8;
  glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-x, -x, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(x, -x, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(x, x, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-x, x, 0.5);
  glEnd();
  
  glFlush();
  glutPostRedisplay();  
}

int main(int argc, char **argv) {
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
  glutInitWindowSize(520, 390);
  glutCreateWindow("Textured Triangles");

#define red {0xff, 0x00, 0x00}
#define yellow {0xff, 0xff, 0x00}
#define magenta {0xff, 0, 0xff}
GLubyte textureData[][3] = {
    red, yellow,
    yellow, red,
};

  glViewport(0, 0, width, height);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glEnable(GL_TEXTURE_2D);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

  glGenTextures(1, &mtexture);
  glBindTexture(GL_TEXTURE_2D, mtexture);
  glTexImage2D(GL_TEXTURE_2D,
               0,                    // level 0
               3,                    // use only R, G, and B components
               2, 2,                 // texture has 2x2 texels
               0,                    // no border
               GL_RGB,               // texels are in RGB format
               GL_UNSIGNED_BYTE,     // color components are unsigned bytes
               textureData);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    glutDisplayFunc(render);
    glutMainLoop();
  return 0;
}
