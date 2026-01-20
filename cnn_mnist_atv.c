/**
 * @file cnn_mnist.c
 * @brief Inferência de uma CNN treinada no MNIST (modelo INT8) no Raspberry Pi Pico W (RP2040) usando TensorFlow Lite Micro.
 *
 * Versão interativa:
 *  - Digite 0..9 no Serial Monitor para testar uma imagem do MNIST correspondente (mnist_samples.h)
 *  - Digite 'a' para rodar teste automático (0..9)
 *  - Digite 'p' para imprimir (ASCII) a imagem selecionada
 *  - Digite 'h' para ajuda
 */

#include <stdio.h>
#include <math.h>
#include <string.h>
#include "pico/stdlib.h"

#include "tflm_wrapper.h"
#include "mnist_samples.h"   // <-- use o arquivo com 10 imagens (0..9)
// Se você ainda não tem mnist_samples.h, gere no Colab e copie para o projeto.

static int argmax_i8(const int8_t* v, int n) {
    int best = 0;
    int8_t bestv = v[0];
    for (int i = 1; i < n; i++) {
        if (v[i] > bestv) { bestv = v[i]; best = i; }
    }
    return best;
}

static int8_t quantize_f32_to_i8(float x, float scale, int zp) {
    long q = lroundf(x / scale) + zp;
    if (q < -128) q = -128;
    if (q >  127) q = 127;
    return (int8_t)q;
}

static void print_help(void) {
    printf("\nComandos:\n");
    printf("  h         -> ajuda\n");
    printf("  0..9      -> roda inferencia na imagem do digito escolhido\n");
    printf("  a         -> teste automatico (0..9)\n");
    printf("  p         -> imprime a imagem atual (ASCII)\n");
    printf("\n");
}

static void print_image_ascii(const uint8_t *img28x28) {
    // Visualizacao simples: ' ' . ':' '*' '#'
    for (int y = 0; y < 28; y++) {
        for (int x = 0; x < 28; x++) {
            uint8_t v = img28x28[y*28 + x];
            char c = ' ';
            if (v > 200) c = '#';
            else if (v > 120) c = '*';
            else if (v > 60)  c = ':';
            else if (v > 20)  c = '.';
            printf("%c", c);
        }
        printf("\n");
    }
}

static void run_inference_on_image(
    const uint8_t *img_u8_28x28,
    int expected_label,
    int8_t *in, int in_bytes,
    int8_t *out, int out_bytes,
    float in_scale, int in_zp,
    float out_scale, int out_zp
) {
    if (in_bytes < 28*28) {
        printf("Erro: input menor que 784 bytes\n");
        return;
    }
    if (out_bytes < 10) {
        printf("Erro: output menor que 10 bytes\n");
        return;
    }

    // Pre-processamento: pixel/255.0 -> quantiza para int8 conforme scale/zp do modelo
    for (int i = 0; i < 28*28; i++) {
        float x = (float)img_u8_28x28[i] / 255.0f;
        in[i] = quantize_f32_to_i8(x, in_scale, in_zp);
    }

    int rc = tflm_invoke();
    if (rc != 0) {
        printf("Invoke falhou: %d\n", rc);
        return;
    }

    int pred = argmax_i8(out, 10);

    printf("\nLabel esperado: %d | Predito: %d\n", expected_label, pred);

    // Scores aproximados (dequant)
    for (int c = 0; c < 10; c++) {
        int8_t q = out[c];
        float y = (float)(q - out_zp) * out_scale;
        printf("c%d: q=%d y~=%f\n", c, (int)q, y);
    }
}

int main() {
    stdio_init_all();
    sleep_ms(1500);
    printf("\n=== MNIST CNN INT8 no Pico W (main em C) - Interativo ===\n");

    int rc = tflm_init();
    if (rc != 0) {
        printf("tflm_init falhou: %d\n", rc);
        while (1) tight_loop_contents();
    }

    printf("Arena usada (bytes): %d\n", tflm_arena_used_bytes());

    int in_bytes = 0;
    int8_t* in = tflm_input_ptr(&in_bytes);

    int out_bytes = 0;
    int8_t* out = tflm_output_ptr(&out_bytes);

    if (!in || !out) {
        printf("Erro: ponteiro input/output nulo\n");
        while (1) tight_loop_contents();
    }

    float in_scale  = tflm_input_scale();
    int   in_zp     = tflm_input_zero_point();
    float out_scale = tflm_output_scale();
    int   out_zp    = tflm_output_zero_point();

    printf("Input bytes: %d | Output bytes: %d\n", in_bytes, out_bytes);
    printf("IN:  scale=%f zp=%d\n", in_scale, in_zp);
    printf("OUT: scale=%f zp=%d\n", out_scale, out_zp);

    // Estado atual: começa no 7 (se existir)
    int current_idx = 7;
    if (current_idx < 0 || current_idx > 9) current_idx = 0;

    print_help();
    printf("Pronto. Digite 0..9 para testar. Exemplo: digite 7 e pressione Enter.\n");

    while (true) {
        int ch = getchar_timeout_us(0);
        if (ch == PICO_ERROR_TIMEOUT) {
            tight_loop_contents();
            continue;
        }

        if (ch == '\r' || ch == '\n') continue;

        if (ch == 'h' || ch == 'H') {
            print_help();
        } else if (ch == 'p' || ch == 'P') {
            printf("\nImagem atual (idx=%d, label=%d):\n", current_idx, mnist_labels[current_idx]);
            print_image_ascii(mnist_images[current_idx]);
        } else if (ch == 'a' || ch == 'A') {
            printf("\nTeste automatico 0..9\n");
            for (int i = 0; i < 10; i++) {
                printf("\n--- Teste idx=%d (label=%d) ---\n", i, mnist_labels[i]);
                run_inference_on_image(
                    mnist_images[i], mnist_labels[i],
                    in, in_bytes, out, out_bytes,
                    in_scale, in_zp, out_scale, out_zp
                );
                sleep_ms(200);
            }
        } else if (ch >= '0' && ch <= '9') {
            current_idx = (int)(ch - '0');
            printf("\n--- Rodando idx=%d (label=%d) ---\n", current_idx, mnist_labels[current_idx]);
            run_inference_on_image(
                mnist_images[current_idx], mnist_labels[current_idx],
                in, in_bytes, out, out_bytes,
                in_scale, in_zp, out_scale, out_zp
            );
        } else {
            printf("\nComando desconhecido '%c'. Digite 'h' para ajuda.\n", (char)ch);
        }
    }
}
