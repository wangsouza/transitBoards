# from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import label_map_util
import tensorflow as tf
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="base path for frozen\
                                            checkpoint detection graph")
ap.add_argument("-l", "--labels", required=True, help="labels file")
ap.add_argument("-i", "--input", required=True, help="path to input output")
ap.add_argument("-o", "--output", required=True, help="path to output video")
ap.add_argument("-n", "--num-classes", type=int, required=True, help="# of\
                                                            class labels")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
                help="minimum probability used to filter weak detections")
args = vars(ap.parse_args())

# Inicia aleatoriamente um conjunto de cores RGB para cada caixa delimitadora.
# A inilialização aleatória é feita por conveniência - Podemos modificar esse
# script para usar cores fixas por rótulo.
COLORS = np.random.uniform(0, 255, size=(args["num_classes"], 3))
# Inicializa o modelo que carregamos no disco.
model = tf.Graph()

# Cria um gerenciador de contexto que torna esse modelo o padrão para execução.
# Carrega a rede serializada do TensorFlow usando os utilitários auxiliares do
# TensorFlow.
with model.as_default():
    # Inicializa a definição do gráfico
    graphDef = tf.GraphDef()

    # Carrega o gráfico a partir do disco.
    with tf.gfile.GFile(args["model"], "rb") as f:
        serializedGraph = f.read()
        graphDef.ParseFromString(serializedGraph)
        tf.import_graph_def(graphDef, name="")

# Carrega a classe de rótulos (classes.pbtxt) a partir do disco.
labelMap = label_map_util.load_labelmap(args["labels"])

# Cria um conjunto de categorias a partir da função
# convert_label_map_to_categories com a opção --num-classes.
categories = label_map_util.convert_label_map_to_categories(
    labelMap, max_num_classes=args["num_classes"],
    use_display_name=True)

# Cria um mapeamento a partir do ID inteiro do rótulo da classe
# (ou seja, o que o TensorFlow retornará ao prever) para o rótulo
# da classe legível por humanos.
categoryIdx = label_map_util.create_category_index(categories)

# Cria uma sessão para executar a inferência.
# Para prever caixas delimitadoras para nossa imagem de entrada,
# primeiro precisamos criar uma sessão TensorFlow e obter
# referências para cada tensor de imagem, caixa delimitadora,
# probabilidade e classes dentro da rede:
with model.as_default():
    with tf.Session(graph=model) as sess:
        # Inicializa os pontos nos arquivos de vídeo.
        stream = cv2.VideoCapture(args["input"])
        writer = None
        # Loop sobre os frames do fluxo de arquivos de vídeo
        while True:
            # Pega o próximo frame
            (grabbed, image) = stream.read()
            # Se o frame não for pego, então nós temos alcançado o final do
            # fluxo.
            if not grabbed:
                break
            # Pega a referência para o tensor de imagem de entrada e o tensor
            # de caixas(boxes)
            # OBS: Essas referências nos permitirão acessar seus valores
            # associados depois de passar a imagem pela rede
            imageTensor = model.get_tensor_by_name("image_tensor:0")
            boxesTensor = model.get_tensor_by_name("detection_boxes:0")

            # Para cada caixa delimitadora nós gostaríamos de saber a pontuação
            # (score), isto é, a probabilidade dos rótulos de classe.
            scoresTensor = model.get_tensor_by_name("detection_scores:0")
            classesTensor = model.get_tensor_by_name("detection_classes:0")
            numDetections = model.get_tensor_by_name("num_detections:0")

            # Pega as dimensões da imagem (W = comprimento e H = altura).
            (H, W) = image.shape[:2]

            # Verifica para ver se nós deveríamos redimensionar junto a
            # comprimento.
            if W > H and W > 1000:
                image = imutils.resize(image, width=1000)
            # Verifica para ver se nós deveríamos redimensionar junto a altura.
            elif H > W and H > 1000:
                image = imutils.resize(image, height=1000)

            # Verifica para ver se nós deveríamos redimensionar junto a altura.
            (H, W) = image.shape[:2]
            output = image.copy()
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)

            # Se o gravador de vídeo for None, inicia a gravação.
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(
                            args["output"],
                            fourcc, 20,
                            (W, H), True)

            # Executar a inferência e calcular as caixas delimitadoras,
            # probabilidades e rótulos de classe. Aqui passamos nossa lista de
            # caixa delimitadora, scores (probabilidades), rótulos de classe e
            # o número de tensores de detecção para o método sess.run.
            # O feed_dict instrui o TensorFlow a definir o imageTensor como a
            # nossa imagem e executar um forward-pass, produzindo nossas caixas
            # delimitadoras, scores e os rótulos de classe.
            (boxes, scores, labels, N) = sess.run(
                                        [boxesTensor, scoresTensor,
                                            classesTensor, numDetections],
                                        feed_dict={imageTensor: image})

            # Achatar as listas em uma única dimensão.
            # As caixas, pontuações e etiquetas são todas matrizes
            # multidimensionais, então as comprimimos em uma matriz 1D,
            # permitindo que passemos facilmente sobre elas.
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            labels = np.squeeze(labels)

            # Loop sobre as predições das caixas delimitadoras
            for (box, score, label) in zip(boxes, scores, labels):
                # Se a probabilidade das predições for menor que o mínimo da
                # confiança, ignore-a.
                if score < args["min_confidence"]:
                    continue

                # Escalona a caixa delimitadora a partir do intervalo [0, 1]
                # para [W, H], ou seja, caminho inverso de quando foi feito o
                # treinamento.
                (startY, startX, endY, endX) = box
                startX = int(startX * W)
                startY = int(startY * H)
                endX = int(endX * W)
                endY = int(endY * H)

                # Desenha a previsão nas imagens de saída.
                # Executa o rótulo legível por humanos no dicionário
                # categoryIdx e desenham o rótulo e a probabilidade associada
                # a imagem.
                label = categoryIdx[label]
                idx = int(label["id"]) - 1
                label = "{}: {:.2f}".format(label["name"], score)
                cv2.rectangle(
                            output,
                            (startX, startY),
                            (endX, endY),
                            COLORS[idx], 2)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(
                            output,
                            label,
                            (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3, COLORS[idx], 1)
            # Grava o arquivo de vídeo no diretório output.
            writer.write(output)
            cv2.imshow("predict traffic sign", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # Fecha os ponteiros dos arquivos de vídeo
        writer.release()
        stream.release()
