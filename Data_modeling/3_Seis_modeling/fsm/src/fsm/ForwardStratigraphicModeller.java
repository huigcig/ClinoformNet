/****************************************************************************
Copyright (c) 2014, Colorado School of Mines and others. All rights reserved.
This program and accompanying materials are made available under the terms of
the Common Public License - v1.0, which accompanies this distribution, and is 
available at http://www.eclipse.org/legal/cpl-v10.html
****************************************************************************/
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

/**
 * Generates realistic stratigraphic models by diffusion.
 * <em>
 * Jacobians of functions used in folding and faulting have been implemented
 * but not tested. Therefore, beware of errors in calculated slopes p2 and p3.
 * </em>
 * @author Xinming Wu
 * @version 2019.01.09
 */
public class ForwardStratigraphicModeller {

  public ForwardStratigraphicModeller(Sampling s1, Sampling s2) {
    _s1 = s1;
    _s2 = s2;
    int n2 = s2.getCount();
    _mask = new boolean[n2];
    _mask[0] = true;
  }

  public ForwardStratigraphicModeller(Sampling s1, Sampling s2, Sampling s3) {
    _s1 = s1;
    _s2 = s2;
    _s3 = s3;
  }

  public void setShoreline(float xsl) {
    _xsl = xsl;
  }

  public void setDiffusionCoefficients(float[][] ks) {
    _ks2 = ks;
  }

  public float[][] getRefModel2d(float[][] ux) {
    int n2 = ux[0].length;
    int n3 = ux.length;
    int m1 = n2;
    setNans2d(0,ux);
    ux = div(ux,max(ux));
    ux = mul(ux,m1);
    Random random = new Random(31);
    float[] r = pow(mul(2.0f,sub(randfloat(random,m1),0.5f)),5.0f);
    //RecursiveExponentialFilter ref = new RecursiveExponentialFilter(1);
    //ref.apply(r,r);
    float[][] rx = new float[n3][n2];
    Sampling s1 = new Sampling(m1);
    SincInterpolator si = new SincInterpolator();
    for (int i3=0; i3<n3; ++i3) {
    for (int i2=0; i2<n2; ++i2) {
      rx[i3][i2] = si.interpolate(s1,r,ux[i3][i2]);
    }}
    return rx;
  }
    
  public float[][][] getRefModel(float[][][] ux) {
    int n1 = ux[0][0].length;
    int n2 = ux[0].length;
    int n3 = ux.length;
    int m1 = n1;
    setNans(0,ux);
    ux = div(ux,max(ux));
    ux = mul(ux,m1);
    Random random = new Random(31);
    float[] r = pow(mul(2.0f,sub(randfloat(random,m1),0.5f)),5.0f);
    //RecursiveExponentialFilter ref = new RecursiveExponentialFilter(1);
    //ref.apply(r,r);
    float[][][] rx = new float[n3][n2][n1];
    Sampling s1 = new Sampling(m1);
    SincInterpolator si = new SincInterpolator();
    for (int i3=0; i3<n3; ++i3) {
    for (int i2=0; i2<n2; ++i2) {
    for (int i1=0; i1<n1; ++i1) {
      rx[i3][i2][i1] = si.interpolate(s1,r,ux[i3][i2][i1]);
    }}}
    /*
    RecursiveGaussianFilterP rgf = new RecursiveGaussianFilterP(1);
    float[][] g1 = new float[n2][n1];
    float[][] g2 = new float[n2][n1];
    rgf.apply1X(ux,g1);
    rgf.applyX1(ux,g2);
    g1 = mul(g1,g1);    g1 = mul(g1,g1);

    g2 = mul(g2,g2);
    float[][] gs = sqrt(add(g1,g2));
    gs = sub(gs,min(gs));
    gs = div(gs,max(gs));
    rx = add(gs,rx);
    */
    return rx;
  }
    
    
  // 2d get reflection model from vp   
  public float[][] getRefModelFromVp2d(float[][] p) {
    int n1 = p[0].length;
    int n2 = p.length;
//     setNans(0.01f,p);
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
    

  public float[][][] getRefModelFromVp(float[][][] p) {
    int n1 = p[0][0].length;
    int n2 = p[0].length;
    int n3 = p.length;
    setNans(0.01f,p);
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
  
  public void setNans2d(float vn, float[][] ux) {
    int n1 = ux[0].length;
    int n2 = ux.length;
    for (int i2=0; i2<n2; ++i2) {
    for (int i1=0; i1<n1; ++i1) {
      if(ux[i2][i1]!=ux[i2][i1])
        ux[i2][i1] = vn;
    }}
  }  
    
  public void setNans(float vn, float[][][] ux) {
    int n1 = ux[0][0].length;
    int n2 = ux[0].length;
    int n3 = ux.length;
    for (int i3=0; i3<n3; ++i3) {
    for (int i2=0; i2<n2; ++i2) {
    for (int i1=0; i1<n1; ++i1) {
      if(ux[i3][i2][i1]!=ux[i3][i2][i1])
        ux[i3][i2][i1] = vn;
    }}}
  }


  public void supply(float sp, float[] hx) {
    int m1 = 11;
    int c1 = (m1-1)/2;
    int cm = c1-1;
    int cp = c1+1;
    float[] px = new float[m1];
    px[cm] = sp;
    px[c1] = sp;
    px[cp] = sp;
    RecursiveExponentialFilter ref = new RecursiveExponentialFilter(1);
    ref.apply(px,px);
    for (int k1=c1; k1<m1; k1++) {
      hx[k1-c1] += px[k1];
    }
  }
    
  public float[][] addWavelet2d(double fpeak, float[][] ux, float[][] rx) {
    double sigma = max(1.0,1.0/(2.0*PI*fpeak));
    int n3 = rx.length;
    int n2 = rx[0].length;
    float[][] q = copy(rx);
    float[][] q1 = new float[n3][n2];
    float[][] q2 = new float[n3][n2];
    float[][] p1 = new float[n3][n2];
    float[][] p2 = new float[n3][n2];
    RecursiveGaussianFilterP rgf1 = new RecursiveGaussianFilterP(1);
    rgf1.apply1X(ux,p1);
    rgf1.applyX1(ux,p2);
    for (int i3=0; i3<n3-1; ++i3) {
    for (int i2=0; i2<n2-1; ++i2) {
      float p1i = p1[i3][i2];
      float p2i = p2[i3][i2];
      float ps = sqrt(p1i*p1i+p2i*p2i);
      if(ps>0) {
        p1[i3][i2] /= ps;
        p2[i3][i2] /= ps;
      }
    }}
    RecursiveGaussianFilterP rgf = new RecursiveGaussianFilterP(sigma);
    for (int id=0; id<2; ++id) { // 2nd directional derivative of Gaussian
      rgf.apply10(q,q1);
      rgf.apply01(q,q2);
      for (int i3=0; i3<n3; ++i3) {
      for (int i2=0; i2<n2; ++i2) {
        float p1i = p1[i3][i2];
        float p2i = p2[i3][i2];
        float q1i = q1[i3][i2];
        float q2i = q2[i3][i2];
        q[i3][i2] = p1i*q1i+p2i*q2i;
      }}
    }
    q = mul(q,-1.0f/rms2d(q)); // negate for Ricker wavelet
    return q;
  }
      
  public float[][][] addWavelet(double fpeak, float[][][] ux, float[][][] rx) {
    double sigma = max(1.0,1.0/(2.0*PI*fpeak));
    int n3 = rx.length;
    int n2 = rx[0].length;
    int n1 = rx[0][0].length;
    float[][][] q = copy(rx);
    float[][][] q1 = new float[n3][n2][n1];
    float[][][] q2 = new float[n3][n2][n1];
    float[][][] q3 = new float[n3][n2][n1];
    float[][][] p1 = new float[n3][n2][n1];
    float[][][] p2 = new float[n3][n2][n1];
    float[][][] p3 = new float[n3][n2][n1];
    RecursiveGaussianFilterP rgf1 = new RecursiveGaussianFilterP(1);
    rgf1.apply1XX(ux,p1);
    rgf1.applyX1X(ux,p2);
    rgf1.applyXX1(ux,p3);
    for (int i3=0; i3<n3-1; ++i3) {
    for (int i2=0; i2<n2-1; ++i2) {
    for (int i1=0; i1<n1-1; ++i1) {
      float p1i = p1[i3][i2][i1];
      float p2i = p2[i3][i2][i1];
      float p3i = p3[i3][i2][i1];
      float ps = sqrt(p1i*p1i+p2i*p2i+p3i*p3i);
      if(ps>0) {
        p1[i3][i2][i1] /= ps;
        p2[i3][i2][i1] /= ps;
        p3[i3][i2][i1] /= ps;
      }
    }}}
    RecursiveGaussianFilterP rgf = new RecursiveGaussianFilterP(sigma);
    for (int id=0; id<2; ++id) { // 2nd directional derivative of Gaussian
      rgf.apply100(q,q1);
      rgf.apply010(q,q2);
      rgf.apply001(q,q3);
      for (int i3=0; i3<n3; ++i3) {
      for (int i2=0; i2<n2; ++i2) {
      for (int i1=0; i1<n1; ++i1) {
        float p1i = p1[i3][i2][i1];
        float p2i = p2[i3][i2][i1];
        float p3i = p3[i3][i2][i1];
        float q1i = q1[i3][i2][i1];
        float q2i = q2[i3][i2][i1];
        float q3i = q3[i3][i2][i1];
        q[i3][i2][i1] = p1i*q1i+p2i*q2i+p3i*q3i;
      }}}
    }
    q = mul(q,-1.0f/rms(q)); // negate for Ricker wavelet
    return q;
  }

  private static float rms2d(float[][] f) {
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
    
  private static float rms(float[][][] f) {
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
    
  //sigma=4,6
  public float[][][] calculateNoise(float sigma, float[][][] fx)
  {
    int m3 = fx.length;
    int m2 = fx[0].length;
    int m1 = fx[0][0].length;
    float[][][] fs = new float[m3][m2][m1];
    float[][][] ns = new float[m3][m2][m1];
    float[][][] fx2 = new float[m3][m2][m1];
    fx2 = mul(fx,1.0f/rms(fx));
    LocalOrientFilter lof = new LocalOrientFilter(4,1,1);
    EigenTensors3 ets = lof.applyForTensors(fx2);
    ets.setEigenvalues(0.01f,1.f,1.f);
    LocalSmoothingFilter lsf = new LocalSmoothingFilter();
    lsf.apply(ets,sigma,fx2,fs);
    ns = sub(fx2,fs);
    return ns;
  }
    
  public float[][][] addRealNoise(float[][][] nx, float[][][] sx) {
    Random rd = new Random();
    int n3 = nx.length;
    int n2 = nx[0].length;
    int n1 = nx[0][0].length;
    int m3 = sx.length;
    int m2 = sx[0].length;
    int m1 = sx[0][0].length;
    int j3 = rd.nextInt(n3-m3);
    int j2 = rd.nextInt(n2-m2);
    int j1 = rd.nextInt(n1-m1);
    float[][][] gx = copy(m1,m2,m3,j1,j2,j3,nx);//从噪声数据中随机切取一块跟干净数据一样大小的区块
    float noise = 0.1f*rd.nextInt(7)+0.2f;//随机定义信噪比
    gx = mul(gx,(float)noise*rms(sx)/rms(gx));//对截取的噪声区块做scale,一方面保证与干净地震数据的幅值一致性，另一方面引入信噪比
    return add(sx,gx);//加入噪声
  }
    
  public float[][] upsample_2d(float d1,float d2, float[][] fx) {
    final int n2 = fx.length;
    final int n1 = fx[0].length;
    final int m2 = round(n2/d2);
    final int m1 = round(n1/d1);
    final Sampling s1 = new Sampling(n1);
    final Sampling s2 = new Sampling(n2);
    final float[][] fr = new float[m2][m1];
    final SincInterpolator si = new SincInterpolator();
    Parallel.loop(m1,new Parallel.LoopInt() {
    public void compute(int i1) {
      for (int i2=0; i2<m2; ++i2) {
        fr[i2][i1] = si.interpolate(s1,s2,fx,i1*d1,i2*d2);
      }
    }});
    return fr;
  }  
    
  public float[][][] upsample_3d(float d1,float d2,float d3, float[][][] fx) {
    final int n3 = fx.length;
    final int n2 = fx[0].length;
    final int n1 = fx[0][0].length;
    final int m3 = round(n3/d3);
    final int m2 = round(n2/d2);
    final int m1 = round(n1/d1);
    final Sampling s1 = new Sampling(n1);
    final Sampling s2 = new Sampling(n2);
    final Sampling s3 = new Sampling(n3);
    final float[][][] fr = new float[m3][m2][m1];
    final SincInterpolator si = new SincInterpolator();
    Parallel.loop(m1,new Parallel.LoopInt() {
    public void compute(int i1) {
      for (int i2=0; i2<m2; ++i2) {
      for (int i3=0; i3<m3; ++i3) {
        fr[i3][i2][i1] = si.interpolate(s1,s2,s3,fx,i1*d1,i2*d2,i3*d3);
      }}
    }});
    return fr;
  }  
    
  public float[][] addRealNoise2d(float[][][] nx, float[][] sx) {
    Random rd = new Random();
    int n3 = nx.length;
    int n2 = nx[0].length;
    int n1 = nx[0][0].length; 
    int m2 = sx.length;
    int m1 = sx[0].length;
    int j3 = rd.nextInt(n3);
    int j2 = rd.nextInt(n2-m2);
    int j1 = rd.nextInt(n1-m1);
    float[][] nx2d = nx[j3];   
    float[][] gx = copy(m1,m2,j1,j2,nx2d);//从噪声数据中随机切取一块跟干净数据一样大小的区块
    float noise = 0.1f*rd.nextInt(7)+0.2f;//随机定义信噪比 
    gx = mul(gx,(float)noise*rms2d(sx)/rms2d(gx));//对截取的噪声区块做scale,一方面保证与干净地震数据的幅值一致性，另一方面引入信噪比
    return add(sx,gx);//加入噪声
  }
    
  public float[][] getShoreLine(float[] p, float theta) {
    int n1 = _s1.getCount();
    int n2 = _s2.getCount();
    float x1 = p[0];
    float x2 = p[1];
    ArrayList<float[]> pa = new ArrayList<>();
    while(x1>=0&&x1<n1&&x2>=0&&x2<n2) {
      x1 += sin(theta);
      x2 += cos(theta);
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

  public float[][] getShelfEdge(float[][] sl) {
    int ns = sl[0].length;
    int[] npl = new int[]{0,1,1,2,2,2,2,3,3,4};
    Random r = new Random();
    int np = npl[r.nextInt(npl.length)];
    if(np==0) {return sl;}
    int s1 = randomSign();
    int pk = round((float)ns/(np+1f));
    float x1t = sl[0][0];
    float x2t = sl[1][0];
    trace("pk="+pk);
    trace("np="+np);
    ArrayList<float[]> sel = new ArrayList<>();
    for (int ip=0; ip<=np; ip++) {
      s1 = -s1;
      float x1i = sl[0][ns-1];
      float x2i = sl[1][ns-1];
      if (ip<np) {
        int d1 = r.nextInt(10)+1;
        int d2 = r.nextInt(10);
        int s2 = randomSign();
        d1 *= s1;
        d2 *= s2;
        int sk = randomSign();
        int pi = pk*(ip+1)+sk*r.nextInt(round(pk*0.4f));
        trace("d1="+d1);
        trace("x1i="+x1i);
        x1i = sl[0][pi]+d1;
        x2i = sl[1][pi]+d2;
      }
      float d1i = x1i-x1t;
      float d2i = x2i-x2t;
      trace("x1i="+x1i);
      trace("x2i="+x2i);
      float dsi = sqrt(d1i*d1i+d2i*d2i);
      for (float di=0f; di<=dsi; di+=1f) {
        float x1 = x1t+di*d1i/dsi;
        float x2 = x2t+di*d2i/dsi;
        sel.add(new float[]{x1,x2});
      }
      x1t = x1i;
      x2t = x2i;
    }

    int ne = sel.size();
    float[][] se = new float[2][ne];
    for (int ie=0; ie<ne; ++ie) {
      float[] xe = sel.get(ie);
      se[0][ie] = xe[0];
      se[1][ie] = xe[1];
    }
    RecursiveExponentialFilter ref = new RecursiveExponentialFilter(8);
    //ref.apply(se[0],se[0]);
    //ref.apply(se[1],se[1]);
    return se;
  }

  private int randomSign() {
    Random r = new Random();
    int sign = r.nextInt(2);
    if (sign==0) sign=-1;
    return sign;
  }


  public void setSurface(float vi, float[][] fx, float[][][] vx) {
    int n3 = vx.length;
    int n2 = vx[0].length;
    int n1 = vx[0][0].length;
    for (int i3=0; i3<n3; ++i3) {
    for (int i2=0; i2<n2; ++i2) {
      float fxi = n1-fx[i3][i2]-1;
      int i1 = round(fxi);
      if(i1>=0&&i1<n1) {
        vx[i3][i2][i1] = vi;
      }
    }}
  }

  public float[][] applyDiffusionX(
    int b3, int e3, float sl, float sp, float sigma, float[][] hx) {
    int n2 = _s2.getCount();
    int n3 = _s3.getCount();
    float[][] ks = new float[n3][n2];
    for (int i3=b3; i3<=e3; i3++) {
      hx[i3][0] += sp;
      hx[i3][1] += sp;
    }
    float[][] ht = copy(hx);
    LocalSmoothingFilter lsf = new LocalSmoothingFilter();
    for (int i3=0; i3<n3; ++i3) {
    for (int i2=0; i2<n2; ++i2) {
      if(hx[i3][i2]>sl) ks[i3][i2] = _ks2[0][0];
      else              ks[i3][i2] = _ks2[0][1];
    }}
    //lsf.applySmoothS(hx,hx);
    lsf.apply(sigma,ks,hx,hx);
    for (int i3=0; i3<n3; ++i3) {
    for (int i2=0; i2<n2; ++i2) {
      hx[i3][i2] = max(hx[i3][i2],_bt3[i3][i2]);
    }}
    return hx;
  }


  public float[] applyDiffusion(float sl, float sp, float sigma, float[] hx) {
    int n2 = _s2.getCount();
    float[] ks = new float[n2];
    hx[0] += sp;
    hx[1] += sp;
    float[] ht = copy(hx);
    LocalSmoothingFilter lsf = new LocalSmoothingFilter();
    for (int i2=0; i2<n2; ++i2) {
      if(i2<sl) ks[i2] = _ks2[0][0];
      else      ks[i2] = _ks2[0][1];
    }
    lsf.apply(sigma,ks,hx,hx);
    for (int i2=0; i2<n2; ++i2) {
      hx[i2] = max(hx[i2],_bt2[i2]);
    }
    //add(hx,_bt2[0]-hx[0],hx);
    return hx;
  }


  private void applyDiffusion(float s, float[] x) {
    for (int it=0; it<10; it++) {
      applyLaplacian(s,copy(x),x);
    }
  }

  private void applyLaplacian(
    float s, float[] x, float[] y){
    s *= 0.25f;
    int n2 = x.length;
    for (int i2=1; i2<n2; ++i2) {
        float xa = 0.0f;
        xa += x[i2  ];
        xa -= x[i2-1];
        xa *= s;
        y[i2-1] -= xa;
        y[i2  ] += xa;
    }
  }

    
  private Sampling _s1,_s2,_s3;
  private float _xsl; //coordinate of the shoreline
  private float[][] _ks2; // diffusion coefficients in 2D
  private boolean[] _mask;
  private float[] _bt2;
  private float[][] _bt3;

  private static void trace(String s) {
    System.out.println(s);
  }
}

