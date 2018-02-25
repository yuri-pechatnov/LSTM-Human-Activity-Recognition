
#ifdef ARDUINO
#include <MPU6050.h>
#include <math.h>
#else
#include <cmath>
#define PROGMEM

struct Vector
{
    union {
        struct {
            float XAxis;
            float YAxis;
            float ZAxis;
        };
        float All[3];
    };
};
#endif

#define forn(x, n) for (int x = 0; x < n; ++x)
#define for3(x) forn(x, 3)

class TGruHarNet {
#include "gru_tensors_declarations.h"
    float State[NHidden];
    Vector SmoothedAcc;

    TGruHarNet() {
        Reset();
    }

    void Reset() {
        for (int i = 0; i < NHidden; ++i) {
            state[i] = 0.0;
        }
        for3 (i) {
            SmoothedAcc.All[i] = 0.0;
        }
    }
    void Add(Vector acc) {
        for3 (i) {
            SmoothedAcc.All[i] = SmoothedAcc.All[i] * AccelSmoothLambda +
                (1 - AccelSmoothLambda) * acc.All[i];
        }

        float input[6] = {acc.XAxis, acc.YAxis, acc.ZAxis, SmoothedAcc.XAxis, SmoothedAcc.YAxis, SmoothedAcc.ZAxis};
    }

};

#include "gru_tensors_definitions.h"
