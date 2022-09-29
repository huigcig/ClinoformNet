package fsm;

import java.awt.*;
import java.util.*;
import javax.swing.*;

import edu.mines.jtk.awt.*;
import edu.mines.jtk.dsp.*;
import edu.mines.jtk.interp.*;
import edu.mines.jtk.mosaic.*;
import edu.mines.jtk.sgl.*;
import edu.mines.jtk.util.*;
import static edu.mines.jtk.util.ArrayMath.*;

public class Depth2TimeConverter {

    /**
   * Returns a velocity model in time domain.
   * @param st time sampling
   * @param sz depth sampling
   * @param vz 3D velocity model in depth domain
   * @return a 3D velocity model in time domain.
   */
  public float[][][][] depth2Time3D(
    Sampling st, Sampling sz, 
    float[][][] vz, float[][][][] pz){
    int np = pz.length;
	  int n3 = vz.length;
	  int n2 = vz[0].length;
	  int nz = vz[0][0].length;
	  int nt = st.getCount();
	  float dz = (float)sz.getDelta();
	  float fz = (float)sz.getFirst();
	  float[] tz = new float[nz];
	  float[] zt = new float[nt];
      float[][][][] pt = new float[np][n3][n2][nt];
      InverseInterpolator ii = new InverseInterpolator(sz,st);
	  for (int i3=0; i3<n3; ++i3) {
	  for (int i2=0; i2<n2; ++i2) {
	    float[] vzi = vz[i3][i2];
	    for (int iz=1; iz<nz; ++iz) {
		    tz[iz] = tz[iz-1]+2f*dz/vzi[iz];
	    }
	    ii.invert(tz,zt);
      for (int ip=0; ip<np; ip++) {
	      _si.interpolate(nz,dz,fz,pz[ip][i3][i2],nt,zt,pt[ip][i3][i2]);
//         if (ip==np-1) {
//           for (int it=0; it<nt; it++) {
//             int iz = sz.indexOfNearest(zt[it]);
//             pt[ip][i3][i2][it] = pz[ip][i3][i2][iz];
//           }
//         }
       
      }
	  }}
    return pt;
  }

  public float[][][] depth2Time2D(
    Sampling st, Sampling sz, 
    float[][] vz, float[][][] pz){
    int np = pz.length;
	  int n2 = vz.length;
	  int nz = vz[0].length;
	  int nt = st.getCount();
	  float dz = (float)sz.getDelta();
	  float fz = (float)sz.getFirst();
	  float[] tz = new float[nz];
	  float[] zt = new float[nt];
      float[][][] pt = new float[np][n2][nt];
      InverseInterpolator ii = new InverseInterpolator(sz,st);
	  for (int i2=0; i2<n2; ++i2) {
	    float[] vzi = vz[i2];
	    for (int iz=1; iz<nz; ++iz) {
		    tz[iz] = tz[iz-1]+2f*dz/vzi[iz];
	    }
	    ii.invert(tz,zt);
      for (int ip=0; ip<np; ip++) {
	      _si.interpolate(nz,dz,fz,pz[ip][i2],nt,zt,pt[ip][i2]);
//         if (ip==np-1) {
//           for (int it=0; it<nt; it++) {
//             int iz = sz.indexOfNearest(zt[it]);
//             pt[ip][i3][i2][it] = pz[ip][i3][i2][iz];
//           }
//         }
       
      }
      }
    return pt;
  }    
  
    
  public float[][] refFromImp2d(float[][] p) {
    int n1 = p[0].length;
    int n2 = p.length;
//     setNans2d(0.01f,p);
    float[][] r = new float[n2][n1];
    for (int i2=0; i2<n2; ++i2) {
      for (int i1=1; i1<n1; ++i1) {
        float pi = p[i2][i1];
        float pm = p[i2][i1-1];
        r[i2][i1] = (pi-pm)/(pi+pm);
      }
    }
    return r;
  }
    
  public static float[][][] refFromImp(float[][][] p) {
    int n1 = p[0][0].length;
    int n2 = p[0].length;
    int n3 = p.length;
    float[][][] r = new float[n3][n2][n1];
    for (int i3=0; i3<n3; ++i3) {
      for (int i2=0; i2<n2; ++i2) {
        for (int i1=1; i1<n1; ++i1) {
          float pi = p[i3][i2][i1];
          float pm = p[i3][i2][i1-1];
          r[i3][i2][i1] = (pi-pm)/(pi+pm);
        }
      }
    }
    return r;    
  }

  public float[][] addWavelet2d(
    float dt, double fpeak, float[][] rx) {
    int n2 = rx.length;
    int n1 = rx[0].length;
    final float[][] sx = new float[n2][n1];
    RickerWavelet rw = new RickerWavelet(fpeak);
    final int kw = 1+(int)(0.5*rw.getWidth()/dt); 
    for (int i2=0; i2<n2; i2++) {
      for (int i1=0; i1<n1; i1++) {
        int b1 = max(i1-kw,0);
        int e1 = min(i1+kw,n1-1);
        for (int k1=b1; k1<=e1; k1++) {
          sx[i2][i1] += rx[i2][k1]*rw.getValue((k1-i1)*dt);
        }
      }
    }
    mul(sx,1.0f/rms2d(sx),sx); // normalization
    return sx;
  }

  public float[][][] addWavelet(
    float dt, double fpeak, float[][][] rx) {
    int n3 = rx.length;
    int n2 = rx[0].length;
    int n1 = rx[0][0].length;
    final float[][][] sx = new float[n3][n2][n1];
    RickerWavelet rw = new RickerWavelet(fpeak);
    final int kw = 1+(int)(0.5*rw.getWidth()/dt);
    Parallel.loop(n3,new Parallel.LoopInt() {
      public void compute(int i3) {
      float[][] sx3 = sx[i3];
      float[][] rx3 = rx[i3];
      for (int i2=0; i2<n2; i2++) {
        for (int i1=0; i1<n1; i1++) {
          int b1 = max(i1-kw,0);
          int e1 = min(i1+kw,n1-1);
          for (int k1=b1; k1<=e1; k1++) {
            sx3[i2][i1] += rx3[i2][k1]*rw.getValue((k1-i1)*dt);
          }
        }
      }
    }});
    mul(sx,1.0f/rms(sx),sx); // normalization
    return sx;
  }

  private float rms(float[][][] f) {
    int n1 = f[0][0].length;
    int n2 = f[0].length;
    int n3 = f.length;
    double sum = 0.0;
    for (int i3=0; i3<n3; ++i3) {
      for (int i2=0; i2<n2; ++i2) {
        for (int i1=0; i1<n1; ++i1) {
          float fi = f[i3][i2][i1];
          sum += fi*fi;
        }
      }
    }
    return (float)sqrt(sum/n1/n2/n3);
  }
    
  private float rms2d(float[][] f) {
    int n1 = f[0].length;
    int n2 = f.length;
    double sum = 0.0;
    for (int i2=0; i2<n2; ++i2) {
      for (int i1=0; i1<n1; ++i1) {
        float fi = f[i2][i1];
        sum += fi*fi;
      }
    }
    return (float)sqrt(sum/n1/n2);
  }

  public class RickerWavelet {
    public RickerWavelet(double fpeak) {
      _a = PI*fpeak;
    }
    public float getValue(double t) {
      double b = _a*t;
      double c = b*b;
      return (float)((1.0-2.0*c)*exp(-c));
    }
    public float getWidth() {
      return (float)(6.0/_a);
    }
    private double _a;
  }

  private static SincInterpolator _si = new SincInterpolator();
  static {
    _si.setExtrapolation(SincInterpolator.Extrapolation.CONSTANT);
  }

}

