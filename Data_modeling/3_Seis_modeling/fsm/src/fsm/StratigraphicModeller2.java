/****************************************************************************
Copyright (c) 2014, Colorado School of Mines and others. All rights reserved.
This program and accompanying materials are made available under the terms of
the Common Public License - v1.0, which accompanies this distribution, and is 
available at http://www.eclipse.org/legal/cpl-v10.html
****************************************************************************/
package fsm;

import java.awt.*;
import java.util.Random;
import java.util.*;
import javax.swing.*;

import edu.mines.jtk.awt.*;
import edu.mines.jtk.dsp.*;
import edu.mines.jtk.interp.*;
import edu.mines.jtk.mosaic.*;
import edu.mines.jtk.sgl.*;
import edu.mines.jtk.util.*;
import static edu.mines.jtk.util.ArrayMath.*;

/**
 * Generates realistic stratigraphic models with random shorelines.
 * @author Xinming Wu
 * @version 2019.10.20
 */
public class StratigraphicModeller2 {

  public StratigraphicModeller2(Sampling s1, Sampling s2) {
    _s1 = s1;
    _s2 = s2;
  }

  public float[][] getModel(float pl, float pr, float[][] se) {
    int np = se[0].length;
    int n1 = _s1.getCount();
    int n2 = _s2.getCount();
    float[][] gx = fillfloat(-2f,n1,n2);
    CubicInterpolator.Method md = CubicInterpolator.Method.MONOTONIC;
    float[][] mk = new float[n2][n1];
    Random r = new Random();
    float[] dp = new float[]{-0.05f,-0.1f,0.0f,0.05f,0.1f};
    for (int ip=0; ip<np-1; ip+=1) {
      /*
      if(se[2][ip]==1f) {
        int idp = r.nextInt(dp.length);
        pr += dp[idp];
      }
      */
      float[] xa  = findBound(-1,pl,se[0][ip],se[1][ip],mk);
      float[] xb  = findBound( 1,pr,se[0][ip],se[1][ip],mk);
      float[] x1l = new float[]{xa[0],se[0][ip]};
      float[] x2l = new float[]{xa[1],se[1][ip]};
      float[] x1r = new float[]{xa[0],se[0][ip],xb[0]};
      float[] x2r = new float[]{xa[1],se[1][ip],xb[1]};
      if(xb[0]-2>se[0][ip]&&xb[1]-4>se[1][ip]) {
         x1r = new float[]{xa[0],se[0][ip],xb[0]-2,xb[0]};
         x2r = new float[]{xa[1],se[1][ip],xb[1]-4,xb[1]};
      }
      float pe = se[0][ip+1]-se[0][ip];
      if (checkMonotonic(x2l)&&pe<pl) {
        int b2 = round(xa[1]);
        int e2 = round(se[1][ip]);
        CubicInterpolator cil = new CubicInterpolator(md,x2l,x1l);
        interpolate(b2,e2,ip,cil,gx);
      }
      if (checkMonotonic(x2r)) {
        int b2 = round(se[1][ip]);
        int e2 = round(xb[1]);
        CubicInterpolator cir = new CubicInterpolator(md,x2r,x1r);
        interpolate(b2,e2,ip,cir,gx);
      }
      if(ip>1) {
        int k1 = round(se[0][ip-1]+1);
        int k2 = round(se[1][ip-1]);
        int p1 = k1+1;
        k1 = min(max(k1,0),n1-1);
        p1 = min(max(p1,0),n1-1);
        k2 = min(max(k2,0),n2-1);
        mk[k2][k1] = 1;
        mk[k2][p1] = 1;
      }
    }
    fillModel(gx);
    gx = sub(max(gx),gx);
    gx = sub(gx,min(gx));
    gx = div(gx,max(gx));
    return gx;
  }

  private boolean checkMonotonic(float[] x2) {
    int np = x2.length;
    for (int ip=1; ip<np; ++ip) {
      if(x2[ip]-x2[ip-1]<=0) return false;
    }
    return true;
  }

  public float[][] getReflectivityModel(float[][] ux) {
    int n2 = ux.length;
    int n1 = ux[0].length;
    int m1 = n1;
    ux = mul(ux,m1);
    Random random = new Random(31);
    float[] r = pow(mul(2.0f,sub(randfloat(random,m1),0.5f)),5.0f);
    //RecursiveExponentialFilter ref = new RecursiveExponentialFilter(1);
    //ref.apply(r,r);
    float[][] rx = new float[n2][n1];
    Sampling s1 = new Sampling(m1);
    SincInterpolator si = new SincInterpolator();
    for (int i2=0; i2<n2; ++i2) {
    for (int i1=0; i1<n1; ++i1) {
      rx[i2][i1] = si.interpolate(s1,r,ux[i2][i1]);
    }}
    RecursiveGaussianFilterP rgf = new RecursiveGaussianFilterP(1);
    float[][] g1 = new float[n2][n1];
    float[][] g2 = new float[n2][n1];
    rgf.apply1X(ux,g1);
    rgf.applyX1(ux,g2);
    g1 = mul(g1,g1);
    g2 = mul(g2,g2);
    float[][] gs = sqrt(add(g1,g2));
    gs = sub(gs,min(gs));
    gs = div(gs,max(gs));
    rx = add(gs,rx);
    return rx;
  }

  public float[][] addWavelet(double fpeak, float[][] ux, float[][] rx) {
    double sigma = max(1.0,1.0/(2.0*PI*fpeak));
    int n2 = rx.length;
    int n1 = rx[0].length;
    float[][] q = copy(rx);
    float[][] q1 = new float[n2][n1];
    float[][] q2 = new float[n2][n1];
    float[][] p1 = new float[n2][n1];
    float[][] p2 = new float[n2][n1];
    RecursiveGaussianFilterP rgf1 = new RecursiveGaussianFilterP(1);
    rgf1.apply1X(ux,p1);
    rgf1.applyX1(ux,p2);
    for (int i2=0; i2<n2-1; ++i2) {
    for (int i1=0; i1<n1-1; ++i1) {
      float ps = 1f/sqrt(p1[i2][i1]*p1[i2][i1]+p2[i2][i1]*p2[i2][i1]);
      p1[i2][i1] *= ps;
      p2[i2][i1] *= ps;
    }}
    RecursiveGaussianFilterP rgf = new RecursiveGaussianFilterP(sigma);
    for (int id=0; id<2; ++id) { // 2nd directional derivative of Gaussian
      rgf.apply10(q,q1);
      rgf.apply01(q,q2);
      for (int i2=0; i2<n2; ++i2) {
        for (int i1=0; i1<n1; ++i1) {
          q[i2][i1] = p1[i2][i1]*q1[i2][i1]+p2[i2][i1]*q2[i2][i1];
        }
      }
    }
    q = mul(q,-1.0f/rms(q)); // negate for Ricker wavelet
    return q;
  }
 
  public float[][] addWavelet(double fpeak, float[][] rx) {
    double sigma = max(1.0,1.0/(2.0*PI*fpeak));
    int n2 = rx[0].length;
    int n1 = rx[0].length;
    float[][] qx = copy(rx);
    RecursiveGaussianFilter rgf = new RecursiveGaussianFilter(sigma);
    for (int id=0; id<2; ++id) { // 2nd directional derivative of Gaussian
      rgf.apply10(qx,qx);
      rgf.apply10(qx,qx);
    }
    qx = mul(qx,-1.0f/rms(qx)); // negate for Ricker wavelet
    return qx;
  }

  private static float rms(float[][] f) {
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



  private void interpolate(
    int b2, int e2, int ip, CubicInterpolator ci, float[][] gx) {
    int n1 = gx[0].length;
    for(int i2=b2; i2<e2; i2++) {
      int i1 = round(ci.interpolate(i2));
      i1 = max(i1,0);
      i1 = min(i1,n1-1);
      gx[i2][i1] = ip+1;
    }
  }

  private void fillModel(float[][] gx) {
    int n2 = gx.length;
    int n1 = gx[0].length;
    for (int i2=0; i2<n2; ++i2) {
    float[] g2 = gx[i2];
    for (int i1=n1-3; i1>=0; --i1) {
      if(g2[i1]<0) {
        g2[i1] = 2*g2[i1+1]-g2[i1+2];
      }
    }}
    checkModel(gx);
  }
  private void checkModel(float[][] gx) {
    int n2 = gx.length;
    int n1 = gx[0].length;
    for (int i2=0; i2<n2; ++i2) {
    float[] g2 = gx[i2];
    for (int i1=n1-2; i1>=0; --i1) {
      if(g2[i1]<g2[i1+1]) {
        g2[i1] = g2[i1+1]+0.01f;
      }
    }}
  }

  private float[] findBound(
    float sign, float p, float x1, float x2,float[][] mk) {
    int n2 = _s2.getCount();
    int n1 = _s1.getCount();
    int k1 = round(x1);
    int k2 = round(x2);
    while(k1>0f&&k1<n1&&k2>0f&&k2<n2&&mk[k2][k1]==0f) {
      x1 += sign*p;
      x2 += sign;
      k1 = round(x1);
      k2 = round(x2);
    }
    return new float[]{x1,x2};
  }

  public float[][] getShoreLine(float[] p, float theta) {
    int n1 = _s1.getCount();
    int n2 = _s2.getCount();
    float x1 = p[0];
    float x2 = p[1];
    ArrayList<float[]> pa = new ArrayList<>();
    while(x1>=0&&x1<n1&&x2>=0&&x2<n2) {
      x1 += 0.1f*sin(theta);
      x2 += 0.1f*cos(theta);
      pa.add(new float[]{x1,x2});
    }
    int np = pa.size();
    float[][] sl = new float[2][np];
    for (int ip=0; ip<np; ++ip) {
      float[] xp = pa.get(ip);
      sl[0][ip] = xp[0];
      sl[1][ip] = xp[1];
    }
    return sl;
  }

  public float[][] getShelfEdgeFromPoints(float[][] sp) {
    int n1 = _s1.getCount();
    int n2 = _s2.getCount();
    int np = sp[0].length;
    float x1t = sp[0][0];
    float x2t = sp[1][0];
    float d = 0.1f;
    ArrayList<float[]> sel = new ArrayList<>();
    for (int ip=1; ip<np; ip++) {
      float x1i = sp[0][ip];
      float x2i = sp[1][ip];
      float d1i = x1i-x1t;
      float d2i = x2i-x2t;
      float dsi = sqrt(d1i*d1i+d2i*d2i);
      for (float di=0f; di<=dsi; di+=d) {
        float x1 = x1t+di*d1i/dsi;
        float x2 = x2t+di*d2i/dsi;
        x1 = max(x1,1f);
        x2 = min(x2,n2-1);
        sel.add(new float[]{x1,x2});
      }
      x1t = x1i;
      x2t = x2i;
    }
    x2t = min(x2t,n2-1);
    while(x1t>=0) {
      x1t -= d;
      sel.add(new float[]{x1t,x2t});
    }
    x1t = max(1,x1t);
    while(x2t<n2) {
      x2t += d;
      sel.add(new float[]{x1t,x2t});
    }
    int ne = sel.size();
    float[][] se = new float[2][ne];
    for (int ie=0; ie<ne; ++ie) {
      float[] xe = sel.get(ie);
      se[0][ie] = xe[0];
      se[1][ie] = xe[1];
    }
    RecursiveExponentialFilter ref = new RecursiveExponentialFilter(20);
    ref.apply(se[0],se[0]);
    ref.apply(se[1],se[1]);
    return se;
  }

  public float[][] getShelfEdge(float[][] sl) {
    int n1 = _s1.getCount();
    int n2 = _s2.getCount();
    int ns = sl[0].length;
    int[] npl = new int[]{0,1,1,2,2,2,2,3,3,4,5,6};
    Random r = new Random();
    int np = npl[r.nextInt(npl.length)];
    int[] kp = new int[np+1];
    float x1t = sl[0][0];
    float x2t = sl[1][0];
    int ct = 0;
    float d = 0.1f;
    ArrayList<float[]> sel = new ArrayList<>();
    while(x1t<n1) {
      x1t += d;
      sel.add(0,new float[]{x1t,1});
      ct++;
    }
    x1t = sl[0][0];
    if(np==0) {
      for (int is=0; is<ns; is++) {
        float x1i = sl[0][is];
        float x2i = sl[1][is];
        x1i = max(x1i,1f);
        x2i = min(x2i,n2-1);
        sel.add(new float[]{x1i,x2i});
        x1t = x1i;
        x2t = x2i;
      }
    } else {
      int s1 = randomSign();
      int pk = round((float)ns/(np+1f));
      for (int ip=0; ip<=np; ip++) {
        kp[ip] = ct;
        s1 = -s1;
        float x1i = sl[0][ns-1];
        float x2i = sl[1][ns-1];
        if (ip<np) {
          int d1 = r.nextInt(25);
          int d2 = r.nextInt(10);
          int s2 = randomSign();
          d1 *= s1;
          d2 *= s2;
          int sk = randomSign();
          int pi = pk*(ip+1)+sk*r.nextInt(round(pk*0.4f));
          x1i = sl[0][pi]+d1;
          x2i = sl[1][pi]+d2;
        }
        float d1i = x1i-x1t;
        float d2i = x2i-x2t;
        float dsi = sqrt(d1i*d1i+d2i*d2i);
        for (float di=0f; di<=dsi; di+=d) {
          float x1 = x1t+di*d1i/dsi;
          float x2 = x2t+di*d2i/dsi;
          x1 = max(x1,1f);
          x2 = min(x2,n2-1);
          sel.add(new float[]{x1,x2});
          ct++;
        }
        x1t = x1i;
        x2t = x2i;
      }
    }
    x2t = min(x2t,n2-1);
    while(x1t>=0) {
      x1t -= d;
      sel.add(new float[]{x1t,x2t});
    }
    x1t = max(1,x1t);
    while(x2t<n2) {
      x2t += d;
      sel.add(new float[]{x1t,x2t});
    }
    int ne = sel.size();
    float[][] se = new float[3][ne];
    for (int ie=0; ie<ne; ++ie) {
      float[] xe = sel.get(ie);
      se[0][ie] = xe[0];
      se[1][ie] = xe[1];
    }
    for (int ip=0; ip<np; ++ip)
      se[2][kp[ip]] = 1;
    RecursiveExponentialFilter ref = new RecursiveExponentialFilter(20);
    ref.apply(se[0],se[0]);
    ref.apply(se[1],se[1]);
    return se;
  }

  private int randomSign() {
    Random r = new Random();
    int sign = r.nextInt(2);
    if (sign==0) sign=-1;
    return sign;
  }

  public float[] getSeaLevel(int ns, float[][] sx) {
    float[] sl = new float[ns];
    CubicInterpolator.Method md = CubicInterpolator.Method.MONOTONIC;
    CubicInterpolator ci = new CubicInterpolator(md,sx[1],sx[0]);
    for (int is=0; is<ns; ++is)
      sl[is] = ci.interpolate(is);
    return sl;
  }



  public void setDiffusionCoefficients(float[][] ks) {
    _ks2 = ks;
  }

  public float[] initializeBasinTopography(float[] x1, float[] x2) {
    int n2 = _s2.getCount();
    float[] bt = new float[n2];
    CubicInterpolator.Method md = CubicInterpolator.Method.MONOTONIC;
    CubicInterpolator ci = new CubicInterpolator(md,x2,x1);
    for (int i2=0; i2<n2; ++i2)
      bt[i2] = ci.interpolate(i2);
    _bt2 = copy(bt);
    return bt;
  }

  private Sampling _s1,_s2;
  private float _xsl; //coordinate of the shoreline
  private float[][] _ks2; // diffusion coefficients in 2D
  private boolean[] _mask;
  private float[] _bt2;
  private float[][] _bt3;

  private static void trace(String s) {
    System.out.println(s);
  }
}
