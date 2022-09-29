package fsm;
import java.nio.*;
import java.io.*;

import edu.mines.jtk.dsp.*;
import edu.mines.jtk.util.Parallel;
import static edu.mines.jtk.util.ArrayMath.*;
import edu.mines.jtk.io.*;
import edu.mines.jtk.util.*;


public class InverseInterpolator {

  public InverseInterpolator(int ni, int no) {
    this(new Sampling(ni),new Sampling(no));
  }

  public InverseInterpolator(Sampling si, Sampling so) {
    Check.argument(si.getCount()>1,"at least two input samples");
    _si = si;
    _so = so;
  }

  public void invert(float[] y, float[] x) {
    int nxi = _si.getCount();
    int nyo = _so.getCount();
    Check.argument(y.length==nxi,"y.length equals number of input samples");
    Check.argument(x.length==nyo,"x.length equals number of output samples");
    int nxim1 = nxi-1;
    int jxi1 = 0;
    int jxi2 = 1;
    float xi1 = (float)_si.getValue(jxi1);
    float xi2 = (float)_si.getValue(jxi2);
    float yi1 = y[jxi1];
    float yi2 = y[jxi2];
    Check.argument(yi1<yi2,"y values strictly increasing");
    float dxody = (xi2-xi1)/(yi2-yi1);
    int jyo = 0;
    float yo = (float)_so.getValue(jyo);
    while (jyo<nyo) {
      if (yo<=yi2 || jxi2==nxim1) {
        x[jyo++] = xi1+(yo-yi1)*dxody;
        if (jyo<nyo)
          yo = (float)_so.getValue(jyo);
      } else if (jxi2<nxim1) {
        xi1 = (float)_si.getValue(++jxi1);
        xi2 = (float)_si.getValue(++jxi2);
        yi1 = y[jxi1];
        yi2 = y[jxi2];
        Check.argument(yi1<yi2,"y values strictly increasing");
        dxody = (xi2-xi1)/(yi2-yi1);
      }
    }
  }
private Sampling _si,_so;
}
