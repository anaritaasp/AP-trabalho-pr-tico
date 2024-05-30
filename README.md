# Classificação de Imagens Volumétricas de Ressonância Magnética

Este repositório contém o código para treinar e avaliar um modelo DenseNet121 para classificar imagens de ressonância magnética (MRI) sintéticas e reais. O projeto é construído utilizando a biblioteca MONAI, PyTorch e outras bibliotecas essenciais para aprendizado profundo e processamento de dados. O código é projetado para lidar com imagens médicas 3D no formato NIfTI (`.nii.gz`).

## Tabela de Conteúdos

- [Visão Geral](#visão-geral)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Requisitos](#requisitos)
- [Preparação do Conjunto de Dados](#preparação-do-conjunto-de-dados)
- [Execução](#execução)
- [Referências](#referências)

## Visão Geral

Este projeto envolve:

1. Preparação e transformação do conjunto de dados de imagens de MRI.
2. Treinamento de um modelo DenseNet121 para classificar imagens como sintéticas (falsas) ou reais.
3. Avaliação do modelo em novas imagens sintéticas.
4. Visualização do processo de treinamento e resultados por meio de vários gráficos.

Devido ao tamanho dos conjuntos de dados, pode ser necessário baixá-los de uma fonte externa. O conjunto de dados pode ser acessado via [WeTransfer](https://we.tl/t-4XkTnA8MI5).

## Estrutura de Diretórios e Ficheiros


```plaintext

|
|__ 50 epochs/              # Diretório com ficheiros dependo no nr epochs usadas
|   |__ 50_DenseNet121      # txt com as informações durante a validação do modelo
|   |__ results50           # txt com os resultados após a função de avaliação
|   |__ "...".png           # gráficos
|
├── original_data/
│   ├── fake/               # Diretório contendo imagens de MRI sintéticas
│   └── real/               # Diretório contendo imagens de MRI reais
|
├── synthetic_image_test/   # Diretório contendo imagens sintéticas para teste
|
|__ classification_model_5  # modelo gerado
|
|__ synthetic_identification # ficheiro python


````

## Requisitos

Antes de executar o código, certifique-se de ter as seguintes bibliotecas instaladas:

- Python >= 3.x
- MONAI
- PyTorch
- Matplotlib
- NumPy

Você pode instalar as dependências executando:

```bash
pip install monai torch matplotlib numpy
```

## Preparação do Conjunto de Dados
Antes do treinamento, organize as imagens de MRI sintéticas e reais nos diretórios original_data/fake e original_data/real, respectivamente. Certifique-se de ter também imagens sintéticas adicionais para avaliação no diretório synthetic_image_test.

## Execução

Para treinar o modelo, execute o script `synthetic_identification.py` com o número desejado de épocas:

```bash
python synthetic_identification.py
```

O programa irá verificar se já existe um modelo treinado com o número de épocas especificado. Dependendo da situação, ele seguirá uma das duas abordagens:

- **Modelo Existente**: Se um modelo treinado com o número de épocas especificado já existir, o script apenas utilizará este modelo e executará a função de avaliação. Os resultados da avaliação serão salvos em um arquivo de texto na pasta <número de épocas> epochs, nomeado como results<número de épocas>.txt.

- **Modelo Inexistente**: Se um modelo com o número de épocas especificado não for encontrado, o script procederá à criação do mesmo para depois podê-lo usar na função de avaliação. Os resultados do treino, incluindo gráficos de precisão e perda, bem como outros ficheiros, serão guardados na pasta número de épocas> epochs.


## Referências
- [MONAI Documentation](https://monai.io/documentation.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)


