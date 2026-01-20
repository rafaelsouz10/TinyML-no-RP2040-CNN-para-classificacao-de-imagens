# 🧠 MNIST CNN INT8 no Raspberry Pi Pico W
## Inferência de Dígitos Manuscritos com TensorFlow Lite Micro (TinyML)

Este projeto demonstra a execução de uma **Rede Neural Convolucional (CNN)** treinada no dataset **MNIST**, quantizada para **INT8**, rodando diretamente em um **Raspberry Pi Pico W (RP2040)** utilizando **TensorFlow Lite Micro**.

O sistema permite **testes interativos via Monitor Serial**, possibilitando selecionar imagens de dígitos (0–9), visualizar a imagem em ASCII e analisar a saída da rede neural em tempo real.

---

## 📋 Características do Projeto

- Modelo CNN treinado no MNIST
- Quantização INT8 (baixo uso de memória)
- Execução embarcada (TinyML)
- Inferência em tempo real no RP2040
- Interface interativa via Serial Monitor
- Visualização ASCII das imagens 28×28
- Teste individual ou automático (0..9)

---

## 🧠 Conceito Geral

O fluxo completo do projeto é:

1. Treinamento da CNN no **Google Colab**
2. Quantização do modelo para **INT8**
3. Exportação do modelo `.tflite` para um **header C**
4. Execução do modelo usando **TensorFlow Lite Micro**
5. Envio de imagens MNIST pré-carregadas para inferência
6. Exibição do resultado no Monitor Serial

---

## 📂 Estrutura do Projeto

```text
cnn_mnist_atv
│
├── cnn_mnist_atv.c              # Código principal (inferência interativa)
├── mnist_samples.h              # Imagens MNIST (0..9) em formato C
├── mnist_cnn_int8_model.h       # Modelo CNN quantizado (INT8)
├── tflm_wrapper.h               # Wrapper para TensorFlow Lite Micro
├── CMakeLists.txt               # Configuração de build (Pico SDK)
├── pico-tflmicro                # git clone https://github.com/raspberrypi/pico-tflmicro.git na raiz do projeto
├── README.md                    # Documentação do projeto
│
└── colab/
    └── CNN_MNIST_ATV.ipynb      # Notebook de treino e quantização (Google Colab)
```
---

## 🖥️ Uso pelo Monitor Serial

Após gravar o firmware, abra o Serial Monitor (115200 baud).

## 📌 Comandos Disponíveis

```text
h         -> Exibe ajuda
0..9      -> Executa inferência no dígito escolhido
a         -> Teste automático (0 até 9)
p         -> Imprime a imagem atual em ASCII
```

---

## 📌 Exemplo de Uso

```text
Digite: 6

Label esperado: 6 | Predito: 6
c0: q=-128 y~=0.000000
c1: q=-128 y~=0.000000
...
c6: q=127  y~=0.996094
...

```
---

## 🔎 Entendendo a Saída da Inferência

Label esperado: rótulo real da imagem MNIST

- Predito: classe escolhida pela CNN
- c0..c9: saída da rede para cada dígito
- q: valor quantizado INT8
- y~: valor aproximado em ponto flutuante (dequantizado)

O dígito com maior valor (argmax) é considerado a predição final.

---

## 🧪 Visualização ASCII da Imagem

Ao pressionar p, a imagem 28×28 é exibida no terminal usando caracteres:

```text
' '  -> fundo
'.'  -> intensidade baixa
':'  -> intensidade média
'*'  -> intensidade alta
'#'  -> pixel forte

Isso ajuda a confirmar visualmente qual dígito está sendo testado.

```
---

## Vídeo Demonstrativo

**Click [AQUI](https://drive.google.com/file/d/1qbE0LZri5XehVuA22vJ8YtHk1VNn_zGE/view?usp=sharing) para acessar o link do Vídeo Ensaio**