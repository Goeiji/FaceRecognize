import cv2

n =input("Digite um número para visualizar a foto: ")
image_path = "fotos/" + n + ".jpg"
img = cv2.imread(image_path)

detectorFace = cv2.CascadeClassifier("original.xml")
reconhecedor = cv2.face.FisherFaceRecognizer_create()
reconhecedor.read("classificadorFisher.yml")
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

cv2.imshow('image', imgResize)
cv2.waitKey(0)
cv2.destroyAllWindows()