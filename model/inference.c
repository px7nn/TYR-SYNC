/*
 * inference.c — WASM inference wrapper for the Tyre Health MLP model.
 * Compiled with emcc. Embeds model weights directly by loading
 * the .mlp file at startup via Emscripten's virtual filesystem.
 * 
    emcc model/inference.c -o static/inference.js -O3 `
    -s EXPORTED_FUNCTIONS="['_init_model','_predict','_malloc','_free']" `
    -s EXPORTED_RUNTIME_METHODS="['ccall','cwrap','getValue','setValue']" `
    -s ALLOW_MEMORY_GROWTH=1 `
    -s MODULARIZE=1 `
    -s EXPORT_NAME="createMLPModule" `
    --preload-file model/tyre_model.mlp@tyre_model.mlp -lm
 */
#include <emscripten/emscripten.h>

#define MLP_IMPLEMENTATION
#define MLP_USE_LIBM
#include "MLP.h"

static Network net = {0};
static int model_loaded = 0;

/* Called from JS after the page loads */
EMSCRIPTEN_KEEPALIVE
int init_model(void) {
    if (model_loaded) return 1;

    Network tmp = {0};
    if (!MLP_Load_Network(&tmp, "tyre_model.mlp")) {
        printf("Failed to load model: %s\n", MLP_ErrorString(MLP_GetLastError()));
        return 0;
    }
    net = tmp;
    model_loaded = 1;
    printf("Model loaded: %zu layers\n", net.n_layers);
    return 1;
}

/*
 * predict() — Run inference on 29 scaled features.
 * input: pointer to 29 doubles (already StandardScaler-transformed)
 * Returns the raw sigmoid output (0..1), caller multiplies by 100.
 */
EMSCRIPTEN_KEEPALIVE
double predict(double *input) {
    if (!model_loaded) return -1.0;

    double output = 0.0;
    MLP_Predict(&net, input, &output);
    return output;
}
