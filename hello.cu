#include <GL/gl.h>
#include <GL/glut.h>


#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

int width = 1024, height = 768;
GLuint mtexture;


void render(){
  //glViewport(0, 0, width, height);

  glColor3f(0,1,0);
  glBegin(GL_TRIANGLES);
  glVertex3f(0.0, 0.0, 0.0);
  glVertex3f(1.0, 0.0, 0.0);
  glVertex3f(0.5, 1.0, 0.0);
  glEnd();  
  
  glutSwapBuffers();
  glutPostRedisplay();
  /*
      glClear(GL_COLOR_BUFFER_BIT);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, mtexture);
  glEnable(GL_TEXTURE_2D);
  glDisable(GL_DEPTH_TEST);
  glDisable(GL_LIGHTING);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-1, -1, -1, -1, -1, -1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glPushAttrib(GL_VIEWPORT_BIT);
  glViewport(0, 0, width, height);
glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
    glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
    glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
    glEnd();  

    glPopAttrib();

    glDisable(GL_TEXTURE_2D);
  */
}

int main(int argc, char **argv) {
  glutInitWindowSize(width, height);
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
  glutInitWindowPosition(50,100);
  glutCreateWindow("example");

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  

  /*

  glGenTextures(1, &mtexture);
  glBindTexture( GL_TEXTURE_2D, mtexture);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
                GL_UNSIGNED_BYTE, NULL);

  glBindTexture(GL_TEXTURE_2D, 0);


  */
    glutDisplayFunc(render);
    glutMainLoop();
  return 0;
}
