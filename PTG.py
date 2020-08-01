import cv2
import datetime

#Imagem 1
n = input("Digite um número para visualizar a foto: ")
image_path = "fotos/" + n + ".jpg"
img = cv2.imread(image_path)

detectorFace = cv2.CascadeClassifier("original.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPH.yml")
largura = 570
altura = 700
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

imgResize = cv2.resize(img, (largura, altura))
imgCinza = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
facesDetectadas = detectorFace.detectMultiScale(imgCinza, scaleFactor=1.3,
                                                            minSize=(200, 200))

for(x, y, l, a) in facesDetectadas:
    imgFace = imgCinza
    id, confianca = reconhecedor.predict(imgFace)
    cv2.rectangle(imgResize, (x, y), (x + l, y + a), (99, 253, 163), 2)
    if id == 1:
        nome = 'Rodrigo'
    else:
        nome = 'Stephen'
    cv2.putText(imgResize, str(nome), (x, y + (a + 30)), font, 2, (99, 253, 163))

i = int(input("Digite 1 para EXIBIR a IMAGEM\n"))
inicial = datetime.datetime.now()
if i == 1:
    print("Entrada:{}\n".format(inicial))
    cv2.imshow('image', imgResize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Fechando")

nome1 = nome

#----------------------------------------------------------------------------------------
#Imagem 2
n = input("Digite um número para visualizar a foto: ")
image_path = "fotos/" + n + ".jpg"
img = cv2.imread(image_path)

detectorFace = cv2.CascadeClassifier("original.xml")
reconhecedor = cv2.face.LBPHFaceRecognizer_create()
reconhecedor.read("classificadorLBPH.yml")
largura = 570
altura = 700
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

imgResize = cv2.resize(img, (largura, altura))
imgCinza = cv2.cvtColor(imgResize, cv2.COLOR_BGR2GRAY)
facesDetectadas = detectorFace.detectMultiScale(imgCinza, scaleFactor=1.3,
                                                            minSize=(200, 200))

for(x, y, l, a) in facesDetectadas:
    imgFace = imgCinza
    id, confianca = reconhecedor.predict(imgFace)
    cv2.rectangle(imgResize, (x, y), (x + l, y + a), (99, 253, 163), 2)
    if id == 1:
        nome = 'Rodrigo'
    else:
        nome = 'Stephen'
    cv2.putText(imgResize, str(nome), (x, y + (a + 30)), font, 2, (99, 253, 163))

i = int(input("Digite 1 para EXIBIR a IMAGEM\n"))
final = datetime.datetime.now()
if i == 1:
    print("Entrada:{}\n".format(final))
    cv2.imshow('image', imgResize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Fechando")

nome2 = nome

#-----------------------------------------------------------------------------------------
#Contar presença
tempodif = (inicial.minute - final.minute) * int(-1)
if(nome1 == nome2) & (tempodif >= 1):
    print("Tempo em Aula: {} minutos".format(tempodif))
    print("Alunos IGUAIS")
    print("Aluno PRESENTE")
else:
    print("Tempo em Aula: {} minutos".format(tempodif))
    print("Alunos DIFERENTES")
    print("Aluno AUSENTE")