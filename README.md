
# OCR com OpenCV e Tesseract

Este projeto tem como objetivo a identificação de textos em imagem de forma satisfatória, utilizando OpenCV e Tesseract OCR para detectar e extrair texto de imagens, aplicando técnicas avançadas de processamento de imagem para melhorar a precisão da leitura.

## 📌 Requisitos

Para executar o projeto, é necessário ter esses requisitos instalados:

- Python
- OpenCV (`cv2`)
- NumPy
- Pytesseract
- Imutils

Você pode instalar as dependências com o seguinte comando:

```sh
pip install opencv-python numpy pytesseract imutils
```
Também é necessário ter o modelo EAST (Efficient and Accurate Scene Text detector) instalado, mas nesse projeto ele já está na pasta ./src

## 🚀 Como executar

1. Certifique-se de que a imagem a ser processada está localizada no diretório `./src/` e ajustada no código (`image = cv2.imread('./src/placarj.webp')`).
2. Execute o script Python:

```sh
python script.py
```

3. O script processará a imagem e imprimirá o texto extraído no console.

## 🔍 Como funciona

O processo de OCR segue as seguintes etapas:

1. **Definição de constantes**
São definidas a configuralção do tesseract a ser utilizada, o caminho para a imagem que será feita a leitura, o caminho para o modelo EAST e as camadas do modelo que serão utilizadas. Segue abaixo o código: 

```python
config_tesseract = '--oem 3 --psm 6'
image = cv2.imread('./src/placarj.webp')

model = "./src/frozen_east_text_detection.pb"
layers = ['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3']
```

2. **Detecção de texto com EAST**
Aqui é onde se define um novo tamanho para a imagem e é mudada para o formato blob, de forma que o modelo consiga fazer a leitura dela. Entçao começa a execução do modelo, usando essa imagem em formado blob como input e então são extraídos os resultados de nível de confiança e localização de textos na imagem, que são definidos como scores e geometry respectivamente:

```python
modelW, modelH = 320,320
image_resized = cv2.resize(image, (modelW, modelH))
blob = cv2.dnn.blobFromImage(image_resized, 1.0, (modelW, modelH), swapRB=True, crop=False)
neural_network = cv2.dnn.readNet(model)
neural_network.setInput(blob)
scores, geometry = neural_network.forward(layers)
```

3. **Definição de variáveis a serem usadas para armazenar e identificar as caixas a serem utilizadas**
Se é extraído a quantidade de linhas e colunas da matriz scores. Cada célula dessa grade contém um valor de confiança sobre a presença de texto naquela região da imagem. É definida a variável que define o nivel minimo de confiança para considerarmos o bounding box, se cria um array de boxes para armazenamento dessas bounding boxes e por fim é criado um array de confidences que armazena o nivel de confiança dessas bounding boxes escolhidas a partir do nivel de confiança:

```python
lines, columns = scores.shape[2:4]
confidence_level = 0.8
boxes = []
confidences = []
```
4. **Processamento dos dados para identificar bounding boxes a serem usados**
Após todas essas definições, é começado o processo para identificar os melhores bounding boxes paraserem utilizar na imagem. Esse processamento dos dados é feito com auxilio de três funções que estão no arquivo './helpres.py':

- geometry_data(geometry, y): extrai os dados de distancia entre o centro da imagem e o topo, direita, base e esquerda da iamgem na respectiva posição y do array geometry. També se é extraído angulo de inclinação.

- calc_geo(data_dtop, data_dright, data_dbottom, data_dleft, data_angle, x, y):  calcula a altura, largura e o angulo de inclinação do bounding box só que na imagem original, baseado nos dados extraídos retornados de geometry_data.

- merge_boxes(boxes, threshold_distance=20, overlap_threshold = 0.1): Identifica bounding boxes que estejam muito proximos ou que estejam tendo sobreposição um com o outro e então é feito o merge deles para serem apenas uma box.

4. **Pré-processamento**: As regiões de interesse (ROI) passam por técnicas como binarização, erosão e dilatação baseadas na valor médio dos pixels encontrados na imagem.

5. **Reconhecimento de texto com Tesseract**: As regiões processadas são passadas para o Tesseract OCR para extração do texto.

6. **Pós-processamento**: Remoção de caracteres indesejados e ajustes na formatação do texto extraído.

## 📂 Estrutura do Projeto

```
.
├── src/
│   ├── placarj.webp  # Imagem de exemplo
│   ├── foto.jpg  # Imagem de exemplo
│   ├── frozen_east_text_detection.pb  # Modelo EAST
├── helpers.py  # Funções auxiliares para cálculos geométricos e fusão de caixas
├── app.py  # Código principal do OCR
└── README.md
```

## 📌 Configuração do Tesseract

Certifique-se de que o Tesseract está instalado e acessível pelo Python. Caso necessário, instale manualmente:

- **Windows**: Baixe e instale a versão do [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki).
- **Linux**: Instale via terminal:

```sh
sudo apt-get install tesseract-ocr
```

Caso o Tesseract não esteja no caminho padrão, defina sua localização no código:

```python
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
```

