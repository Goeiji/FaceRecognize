import cv2
import os
import numpy as np

eigenface = cv2.face.EigenFaceRecognizer_create()
fisherface = cv2.face.FisherFaceRecognizer_create()
lbph = cv2.face.LBPHFaceRecognizer_create()

def getImagemComId():
    largura = 570
    altura = 700

    caminhos = [os.path.join('fotos', f)
                for f in os.listdir('fotos')]
    #print(caminhos)
    faces = []
    ids = []
    for caminhoImagem in caminhos:
        imagemFace = cv2.cvtColor(cv2.imread(caminhoImagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminhoImagem)[-1].split('.')[0])
        #"[-1]" retorna o último elemento do path, ou seja, as fotos
        #"[0]" se refere ao index da parte do split que eu desejo pegar
        #cv2.imshow("Face", imagemFace)
        #cv2.waitKey(10)
        ids.append(id)
        faces.append(cv2.resize(imagemFace, (largura, altura)))
    return np.array(ids), faces

ids, faces = getImagemComId()

print("Treinando...")
eigenface.train(faces, ids)
eigenface.write('classificadorEigen.yml')

fisherface.train(faces, ids)
fisherface.write('classificadorFisher.yml')

lbph.train(faces, ids)
lbph.write('classificadorLBPH.yml')

print("Treinamento COMPLETO")